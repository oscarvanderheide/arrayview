"""VS Code browser opening, remote/tunnel handling, and SSH guidance."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

from arrayview._session import _vprint
from arrayview._platform import _in_vscode_terminal, _is_vscode_remote
from arrayview._vscode_extension import _configure_vscode_port_preview, _ensure_vscode_extension, _VSCODE_EXT_FRESH_INSTALL
from arrayview._vscode_signal import _open_via_signal_file, _open_direct_via_signal_file, _schedule_remote_open_retries

# Whether the "set port to Public" message has been printed this session.
_remote_message_shown = False
_ssh_message_shown = False


# ---------------------------------------------------------------------------
# Browser opening
# ---------------------------------------------------------------------------


def _print_viewer_location(url: str) -> None:
    """Print a viewer location hint (verbose only)."""
    if not _is_vscode_remote():
        _vprint(f"[ArrayView] {url}", flush=True)


def _open_browser(
    url: str,
    blocking: bool = False,
    force_vscode: bool = False,
    title: str | None = None,
    filepath: str | None = None,
    floating: bool = False,
) -> None:
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

    def _do() -> None:
        in_vscode = _in_vscode_terminal()
        is_remote = _is_vscode_remote()
        opened = False

        try:
            parsed_port = int(url.split(":")[2].split("/")[0].split("?")[0])
        except Exception:
            parsed_port = 8000

        if is_remote:
            # Remote/tunnel: install extension + write signal file.
            ext_ok = _ensure_vscode_extension()
            if ext_ok and _VSCODE_EXT_FRESH_INSTALL:
                time.sleep(1.5)

            if filepath:
                # Direct webview mode: no ports, no WebSocket, no auth needed.
                # The extension spawns a Python subprocess and bridges via
                # postMessage — completely bypasses the port-forwarding issue.
                _open_direct_via_signal_file(filepath, title=title, floating=floating)
                _vprint(
                    "[ArrayView] Remote tunnel → direct webview mode (no port needed)",
                    flush=True,
                )
            else:
                # Fallback to URL-based mode (e.g. --serve, or relay)
                global _remote_message_shown
                if not _remote_message_shown:
                    _remote_message_shown = True
                    print(
                        f"[ArrayView] Remote tunnel session on port {parsed_port}.\n"
                        f"  VS Code Ports tab: right-click port {parsed_port} → Port Visibility → Public.\n"
                        f"  If the viewer tab shows an auth page, make the port Public then reload the tab.",
                        flush=True,
                    )
                _open_via_signal_file(url, title=title, floating=floating)
                _schedule_remote_open_retries(url, interval=10.0, count=2)
            if not ext_ok:
                _vprint(
                    "[ArrayView] extension install could not be verified — signal file written anyway",
                    flush=True,
                )
            return

        if force_vscode or in_vscode:
            # Local VS Code terminal (or --window vscode forced): install extension + signal file.
            _configure_vscode_port_preview(parsed_port)
            ext_ok = _ensure_vscode_extension()
            if ext_ok and _VSCODE_EXT_FRESH_INSTALL:
                time.sleep(1.5)
            # Always write the signal file: the extension may already be
            # installed even if _ensure failed (e.g. `code` CLI not found).
            _open_via_signal_file(url, title=title, floating=floating)
            # Schedule a retry in case the extension was mid-reload when the
            # first signal was written (e.g. first run with old version removed).
            _schedule_remote_open_retries(url, interval=10.0, count=2)
            opened = True

        is_plain_ssh = (
            not is_remote
            and not in_vscode
            and bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"))
        )
        if is_plain_ssh:
            global _ssh_message_shown
            if not _ssh_message_shown:
                _ssh_message_shown = True
                try:
                    port_hint = int(url.split(":")[2].split("/")[0].split("?")[0])
                except Exception:
                    port_hint = parsed_port
                hostname = os.uname().nodename or "<this-host>"
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

        # Local sessions still benefit from a clickable terminal URL.
        _vprint(f"\n  \033[1;36m→ {url}\033[0m\n", flush=True)

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()
