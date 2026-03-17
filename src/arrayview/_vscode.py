"""VS Code extension management, signal-file IPC, and browser opening."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from importlib.resources import files as _pkg_files

from arrayview._session import _vprint
from arrayview._platform import (
    _find_code_cli,
    _find_vscode_ipc_hook,
    _in_vscode_terminal,
    _is_vscode_remote,
)

# Whether the "set port to Public" message has been printed this session.
_remote_message_shown = False


# ---------------------------------------------------------------------------
# VS Code .app bundle detection (macOS)
# ---------------------------------------------------------------------------


def _vscode_app_bundle() -> str | None:
    """Return the path to the VS Code .app bundle on macOS, derived from the code CLI."""
    code = _find_code_cli()
    if not code:
        return None
    try:
        real = os.path.realpath(code)
        idx = real.find(".app")
        if idx != -1:
            return real[: idx + 4]
    except Exception:
        pass
    for candidate in [
        "/Applications/Visual Studio Code.app",
        os.path.expanduser("~/Applications/Visual Studio Code.app"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------

_VSCODE_EXT_INSTALLED = False  # cached so we only check once per process
_VSCODE_EXT_FRESH_INSTALL = False  # True if we just installed it this session
_VSCODE_EXT_VERSION = "0.9.16"  # must match vscode-extension/package.json
_VSCODE_SIGNAL_FILENAME = "open-request-v0900.json"
_VSCODE_COMPAT_SIGNAL_FILENAMES: tuple[str, ...] = ("open-request-v0800.json",)
_VSCODE_PORT_SETTINGS_SETTLE_SECONDS = 2.0
_VSCODE_SIGNAL_MAX_AGE_MS = (
    60_000  # 60s: survive extension-host reloads (~12s) plus simpleBrowser.show latency
)


# ---------------------------------------------------------------------------
# Extension install / metadata
# ---------------------------------------------------------------------------


def _bundled_vscode_vsix_version(vsix_path: str) -> str | None:
    """Return the bundled opener extension version recorded inside the VSIX."""
    try:
        with zipfile.ZipFile(vsix_path) as zf:
            with zf.open("extension/package.json") as f:
                data = json.load(f)
        version = data.get("version")
        return version if isinstance(version, str) else None
    except Exception as exc:
        _vprint(
            f"[ArrayView] could not inspect VSIX version at {vsix_path}: {exc}",
            flush=True,
        )
        return None


def _patch_vscode_extension_metadata(version: str) -> None:
    """Remove broken targetPlatform metadata written by VS Code for local VSIX installs."""
    for base_dir in (
        os.path.expanduser("~/.vscode-server/extensions"),
        os.path.expanduser("~/.vscode/extensions"),
    ):
        package_json = os.path.join(
            base_dir, f"arrayview.arrayview-opener-{version}", "package.json"
        )
        if not os.path.isfile(package_json):
            continue
        try:
            with open(package_json) as f:
                data = json.load(f)
            metadata = data.get("__metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("targetPlatform") == "undefined"
            ):
                del metadata["targetPlatform"]
                with open(package_json, "w") as f:
                    json.dump(data, f, indent=8)
                    f.write("\n")
        except Exception as exc:
            _vprint(
                f"[ArrayView] could not patch extension metadata at {package_json}: {exc}",
                flush=True,
            )


def _remove_old_extension_versions(current_version: str) -> None:
    """Delete extension directories for versions older than current_version.

    When multiple versions of arrayview-opener are installed side-by-side,
    VS Code may load an older version instead of the latest.  Removing stale
    directories ensures the correct version is picked up on the next reload.
    """
    import shutil

    for ext_base in (
        os.path.expanduser("~/.vscode-server/extensions"),
        os.path.expanduser("~/.vscode/extensions"),
    ):
        if not os.path.isdir(ext_base):
            continue
        try:
            entries = os.listdir(ext_base)
        except OSError:
            continue
        prefix = "arrayview.arrayview-opener-"
        for entry in entries:
            if not entry.startswith(prefix):
                continue
            version_str = entry[len(prefix) :]
            if version_str == current_version:
                continue  # keep
            old_dir = os.path.join(ext_base, entry)
            try:
                shutil.rmtree(old_dir)
                _vprint(f"[ArrayView] removed old extension: {entry}", flush=True)
            except Exception as exc:
                _vprint(f"[ArrayView] could not remove {entry}: {exc}", flush=True)


def _extension_on_disk(version: str) -> bool:
    """Return True if the extension directory for *version* exists on disk."""
    for base in (
        os.path.expanduser("~/.vscode/extensions"),
        os.path.expanduser("~/.vscode-server/extensions"),
    ):
        if os.path.isdir(os.path.join(base, f"arrayview.arrayview-opener-{version}")):
            return True
    return False


def _ensure_vscode_extension() -> bool:
    """Install the bundled arrayview-opener VS Code extension for local VS Code use.

    The extension bridges local VS Code terminals to ``simpleBrowser.show(...)``
    and, in remote/tunnel sessions, can actively invoke VS Code's forwarded-port
    commands to promote the port to public preview.

    Skips reinstallation if the correct version is already present on disk —
    reinstalling with --force causes an extension-host reload, which creates a
    timing gap during which a signal file may be missed.  Old extension
    directories are cleaned up before any fresh install.
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    if _VSCODE_EXT_INSTALLED:
        return True

    # Fast path: correct version already installed — no reinstall needed.
    # Reinstalling with --force triggers an extension-host reload, which
    # creates a ~10-15s gap during which the signal file can be missed.
    _remove_old_extension_versions(_VSCODE_EXT_VERSION)
    if _extension_on_disk(_VSCODE_EXT_VERSION):
        _VSCODE_EXT_INSTALLED = True
        _vprint(
            f"[ArrayView] extension v{_VSCODE_EXT_VERSION} already installed — skipping reinstall",
            flush=True,
        )
        return True

    code = _find_code_cli()
    if not code:
        return False

    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc and "VSCODE_IPC_HOOK_CLI" not in env:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    vsix_path = str(_pkg_files("arrayview").joinpath("arrayview-opener.vsix"))
    if not os.path.isfile(vsix_path):
        return False
    bundled_version = _bundled_vscode_vsix_version(vsix_path)
    if bundled_version != _VSCODE_EXT_VERSION:
        _vprint(
            f"[ArrayView] extension version mismatch: bundled={bundled_version!r} "
            f"expected={_VSCODE_EXT_VERSION!r} — rebuild arrayview-opener.vsix",
            flush=True,
        )
        return False

    try:
        r = subprocess.run(
            [code, "--install-extension", vsix_path, "--force"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        combined = (r.stdout or "") + (r.stderr or "")
        install_failed = (
            "Cannot install" in combined
            or "Failed Installing Extensions" in combined
            or "extension/package.json not found inside zip" in combined
            or "Error:" in combined
        )
        if r.returncode == 0 and not install_failed:
            _patch_vscode_extension_metadata(_VSCODE_EXT_VERSION)
            _VSCODE_EXT_INSTALLED = True
            _VSCODE_EXT_FRESH_INSTALL = True
            return True
        print(f"[ArrayView] extension install failed: {combined.strip()!r}", flush=True)
    except Exception as exc:
        print(f"[ArrayView] extension install error: {exc}", flush=True)
    return False


# ---------------------------------------------------------------------------
# Port settings
# ---------------------------------------------------------------------------


def _configure_vscode_port_preview(port: int) -> bool:
    """Write VS Code port settings for the arrayview server.

    In VS Code remote/tunnel sessions this writes both Machine and User
    settings files to maximize the chance VS Code honors them. Workspace-
    level settings are unreliable for port privacy in tunnel sessions.

    In local VS Code terminals we keep the workspace-level attribute so
    auto-forward/silent remains configured when relevant.

    Returns True on success.
    """

    def _strip_json_comments(raw: str) -> str:
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
        raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.MULTILINE)
        return raw

    def _load_settings(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = f.read()
            cleaned = _strip_json_comments(raw)
            return json.loads(cleaned) if cleaned.strip() else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_settings(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        settings = _load_settings(path)
        attrs = settings.setdefault("remote.portsAttributes", {})
        attrs[str(port)] = {
            "protocol": "http",
            "label": "ArrayView",
            "onAutoForward": "silent",
            "privacy": "public",
        }
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")

    try:
        in_vscode = _in_vscode_terminal()
        is_remote = _is_vscode_remote()

        if is_remote:
            home = os.path.expanduser("~")
            targets: list[str] = []
            for root in (
                os.path.join(home, ".vscode", "cli"),
                os.path.join(home, ".vscode-server"),
            ):
                if os.path.isdir(root):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))
            if not targets:
                # Fallback: write to the most common paths even if root
                # directories don't exist yet.
                for root in (
                    os.path.join(home, ".vscode-server"),
                    os.path.join(home, ".vscode", "cli"),
                ):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))

            for settings_path in targets:
                _write_settings(settings_path)
            return True

        if in_vscode:
            settings_path = os.path.join(os.getcwd(), ".vscode", "settings.json")
            _write_settings(settings_path)
        return True
    except Exception as exc:
        _vprint(f"[ArrayView] could not write port settings: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Signal-file IPC
# ---------------------------------------------------------------------------


def _open_via_signal_file(
    url: str, delay: float = 0.0, title: str | None = None
) -> bool:
    """Write the URL to the versioned ArrayView opener signal file.

    *title* is shown as the VS Code tab label (e.g. "ArrayView: sample.npy").
    If omitted it is auto-derived from the session name embedded in the URL's
    ``?sid=`` query parameter.
    """
    if title is None:
        try:
            import urllib.parse as _up

            _sid = _up.parse_qs(_up.urlparse(url).query).get("sid", [None])[0]
            if _sid:
                import arrayview._session as _sm

                _sess = _sm.SESSIONS.get(_sid)
                if _sess:
                    title = f"ArrayView: {_sess.name}"
        except Exception:
            pass

    payload: dict = {
        "action": "open-preview",
        "url": url,
        "maxAgeMs": _VSCODE_SIGNAL_MAX_AGE_MS,
    }
    if title:
        payload["title"] = title
    return _write_vscode_signal(payload, delay=delay)


def _schedule_remote_open_retries(
    url: str, interval: float = 15.0, count: int = 2
) -> None:
    """Backup retries via signal file (extension handles primary retries internally).

    Reduced to 2 retries at 15s intervals.  The VS Code extension now retries
    ``simpleBrowser.show()`` internally after claiming a signal, so Python-side
    retries are only needed as a Safety net (e.g. extension not loaded yet after
    a fresh install).
    """
    import urllib.parse as _urlparse

    _parsed = _urlparse.urlparse(url)
    _qs = _urlparse.parse_qs(_parsed.query)
    _target_sid = _qs.get("sid", [None])[0]

    def _loop() -> None:
        for i in range(count):
            time.sleep(interval)
            if _target_sid:
                import arrayview._session as _sm

                if _target_sid in _sm.VIEWER_SIDS:
                    return  # this session's viewer connected
            _open_via_signal_file(url)

    threading.Thread(target=_loop, daemon=True).start()


def _write_vscode_signal(payload: dict, delay: float = 0.0) -> bool:
    """Write a versioned control payload for the VS Code opener extension.

    Per-window targeting: when VSCODE_IPC_HOOK_CLI can be found (directly or
    via parent-process walk), Python computes hookTag = SHA-256(ipc_hook)[:16]
    and writes *only* to the targeted file
    ``~/.arrayview/open-request-ipc-<hookTag>.json``.  The extension in the
    correct VS Code window checks that same file and claims it exclusively.

    The registration-file check (window-<tag>.json) has been intentionally
    removed: writing to the targeted file is safe even before the extension
    has activated, because the extension calls tryOpenSignalFile() immediately
    on activate and will find the file.  Removing the check eliminates the
    race window where Python fell back to the shared file (claimable by any
    window) just because the extension hadn't written its registration yet.

    Falls back to the shared signal file only when ipc_hook is not found at
    all (e.g. non-VS Code environment or fully-stripped env).
    """
    import hashlib
    from arrayview._platform import _find_vscode_ipc_hook

    signal_dir = os.path.expanduser("~/.arrayview")
    try:
        os.makedirs(signal_dir, exist_ok=True)
        if delay > 0:
            time.sleep(delay)
        data = dict(payload)
        data.setdefault("sentAtMs", int(time.time() * 1000))
        data.setdefault("maxAgeMs", _VSCODE_SIGNAL_MAX_AGE_MS)
        data.setdefault("requestId", uuid.uuid4().hex)

        # Always use the targeted file when we have an IPC hook — no
        # registration-file check.  This prevents any other VS Code window
        # from racing to claim the shared fallback file.
        ipc_hook = _find_vscode_ipc_hook()
        if ipc_hook:
            own_tag = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
            data["hookTag"] = (
                own_tag  # extension verifies this on shared-fallback claims
            )
            filenames: tuple[str, ...] = (f"open-request-ipc-{own_tag}.json",)
        else:
            # Fallback: shared signal file (any window's extension may claim it)
            filenames = (_VSCODE_SIGNAL_FILENAME, *_VSCODE_COMPAT_SIGNAL_FILENAMES)

        for filename in filenames:
            try:
                os.unlink(os.path.join(signal_dir, filename))
            except FileNotFoundError:
                pass
        for filename in filenames:
            signal_file = os.path.join(signal_dir, filename)
            tmp_file = signal_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            os.replace(tmp_file, signal_file)  # atomic on POSIX + Windows
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Browser opening
# ---------------------------------------------------------------------------


def _print_viewer_location(url: str) -> None:
    """Print a viewer location hint."""
    if not _is_vscode_remote():
        print(f"[ArrayView] {url}", flush=True)


def _open_browser(url: str, blocking: bool = False, force_vscode: bool = False) -> None:
    """Open *url* locally, or configure VS Code remote auto-preview behavior.

    Strategy (see log.txt for what was tried and why):
    1. Remote VS Code terminal:
       a. Configure the port as ``silent`` and ``public`` in
          ``remote.portsAttributes``.
       b. Write the signal file; the workspace extension converts the URL via
          asExternalUri and opens Simple Browser in the local VS Code client.
    2. Local VS Code terminal (or force_vscode=True):
       a. Install the helper extension.
       b. Write the signal file so the extension opens Simple Browser locally.
     3. Fallback: open/xdg-open with the http URL (system browser).
     4. Always print the URL.
    """

    def _do() -> None:
        ipc = _find_vscode_ipc_hook()
        is_remote = _is_vscode_remote()
        opened = False

        try:
            parsed_port = int(url.split(":")[2].split("/")[0].split("?")[0])
        except Exception:
            parsed_port = 8000

        if is_remote:
            # Remote/tunnel: install extension + write signal file.
            # The workspace extension resolves the devtunnel URL via asExternalUri
            # and opens Simple Browser with ?sid= preserved.
            # Port visibility must be Public (user sets it manually in Ports tab,
            # or it was pre-configured by `arrayview --serve`).
            # NOTE: Do NOT call _configure_vscode_port_preview() here — writing
            # settings files at open-time can disrupt active port forwarding.
            # The --serve path and CLI new-server path handle it at setup time.
            global _remote_message_shown
            if not _remote_message_shown:
                _remote_message_shown = True
                print(
                    f"[ArrayView] Remote tunnel session on port {parsed_port}.\n"
                    f"  VS Code Ports tab: right-click port {parsed_port} → Port Visibility → Public.\n"
                    f"  If the Simple Browser tab shows an auth page, make the port Public then reload the tab.",
                    flush=True,
                )
            ext_ok = _ensure_vscode_extension()
            if ext_ok and _VSCODE_EXT_FRESH_INSTALL:
                time.sleep(1.5)
            # Always write the signal file: the extension may already be
            # installed from a prior session even if _ensure failed (e.g.
            # `code` CLI not in PATH, Julia stripped env, etc.).
            _open_via_signal_file(url)
            # Note: retries deliberately omitted. simpleBrowser.show() opens a
            # NEW tab on every call (not idempotent). If the port was Private and
            # shows an auth page, the user should make the port Public and reload
            # the Simple Browser tab manually.
            if not ext_ok:
                _vprint(
                    "[ArrayView] extension install could not be verified — signal file written anyway",
                    flush=True,
                )
            return

        if force_vscode or ipc or _in_vscode_terminal():
            # Local VS Code terminal (or --window vscode forced): install extension + signal file.
            _configure_vscode_port_preview(parsed_port)
            ext_ok = _ensure_vscode_extension()
            if ext_ok and _VSCODE_EXT_FRESH_INSTALL:
                time.sleep(1.5)
            # Always write the signal file: the extension may already be
            # installed even if _ensure failed (e.g. `code` CLI not found).
            _open_via_signal_file(url)
            # Schedule a retry in case the extension was mid-reload when the
            # first signal was written (e.g. first run with old version removed).
            _schedule_remote_open_retries(url, interval=10.0, count=2)
            opened = True

        is_plain_ssh = (
            not is_remote
            and not ipc
            and not _in_vscode_terminal()
            and bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"))
        )
        if is_plain_ssh:
            try:
                port_hint = int(url.split(":")[2].split("/")[0].split("?")[0])
            except Exception:
                port_hint = parsed_port
            print(
                f"[ArrayView] SSH session — to view this array remotely:\n"
                f"\n"
                f"  Option A — simple port-forward (direct SSH access):\n"
                f"    ssh -L {port_hint}:localhost:{port_hint} <user>@<this-host>\n"
                f"    Then open: http://localhost:{port_hint}/\n"
                f"\n"
                f"  Option B — relay through an existing public ArrayView server\n"
                f"  (best for multi-hop: local → tunnel-remote → this host):\n"
                f"    1. Re-connect with a reverse tunnel (pick any free local port, e.g. 8765):\n"
                f"       ssh -R 8765:localhost:8000 <user>@<this-host>\n"
                f"    2. Then run:\n"
                f"       arrayview <file> --relay 8765\n"
                f"    The array is sent to the relay server on port 8000 of the\n"
                f"    intermediate host and opens in Simple Browser automatically.\n",
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
        print(f"\n  \033[1;36m→ {url}\033[0m\n", flush=True)

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()
