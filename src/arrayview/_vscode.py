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
_ssh_message_shown = False


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
_VSCODE_EXT_VERSION = "0.14.1"  # current bundled extension version
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


def _extension_on_disk(version: str, vsix_path: str | None = None) -> bool:
    """Return True if the extension directory for *version* exists on disk.

    When *vsix_path* is given, also verifies that the installed extension
    matches the bundled VSIX by comparing a content hash stored at install
    time.  This catches rebuilds during development where the version stays
    the same but the VSIX content changed.
    """
    import hashlib

    for base in (
        os.path.expanduser("~/.vscode/extensions"),
        os.path.expanduser("~/.vscode-server/extensions"),
    ):
        ext_dir = os.path.join(base, f"arrayview.arrayview-opener-{version}")
        if not os.path.isdir(ext_dir):
            continue
        if vsix_path is None:
            return True
        # Compare content hash
        try:
            vsix_hash = hashlib.md5(open(vsix_path, "rb").read()).hexdigest()
        except OSError:
            return True  # can't read VSIX, assume installed is fine
        hash_file = os.path.join(ext_dir, ".vsix_hash")
        try:
            installed_hash = open(hash_file).read().strip()
        except OSError:
            installed_hash = None
        if installed_hash == vsix_hash:
            return True
        _vprint(
            f"[ArrayView] VSIX content changed (installed={installed_hash}, bundled={vsix_hash}) — reinstalling",
            flush=True,
        )
        return False
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

    The authoritative version is read from the bundled VSIX — no hardcoded
    version constant needed.
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    if _VSCODE_EXT_INSTALLED:
        return True

    vsix_path = str(_pkg_files("arrayview").joinpath("arrayview-opener.vsix"))
    if not os.path.isfile(vsix_path):
        return False
    ext_version = _bundled_vscode_vsix_version(vsix_path)
    if not ext_version:
        _vprint("[ArrayView] could not read version from bundled VSIX", flush=True)
        return False

    # Fast path: correct version and content already installed — no reinstall
    # needed.  Reinstalling with --force triggers an extension-host reload,
    # which creates a ~10-15s gap during which the signal file can be missed.
    _remove_old_extension_versions(ext_version)
    if _extension_on_disk(ext_version, vsix_path):
        _VSCODE_EXT_INSTALLED = True
        _vprint(
            f"[ArrayView] extension v{ext_version} already installed — skipping reinstall",
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
            _patch_vscode_extension_metadata(ext_version)
            # Write content hash so future runs can detect VSIX rebuilds
            # without a version bump (common during development).
            try:
                import hashlib
                vsix_hash = hashlib.md5(open(vsix_path, "rb").read()).hexdigest()
                for base in (
                    os.path.expanduser("~/.vscode/extensions"),
                    os.path.expanduser("~/.vscode-server/extensions"),
                ):
                    ext_dir = os.path.join(base, f"arrayview.arrayview-opener-{ext_version}")
                    if os.path.isdir(ext_dir):
                        with open(os.path.join(ext_dir, ".vsix_hash"), "w") as f:
                            f.write(vsix_hash)
            except Exception:
                pass  # non-critical
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

    # Helper: get parent PID of a process (macOS/Linux)
    def _ppid(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True,
                text=True,
                timeout=1,
            )
            return int(r.stdout.strip())
        except Exception:
            return -1

    # Collect our ancestor PIDs with depth (depth=1 means direct parent)
    our_ancestors: dict[int, int] = {}
    cur = os.getpid()
    for depth in range(1, 25):
        p = _ppid(cur)
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

    # Walk ancestors looking for ARRAYVIEW_WINDOW_ID (uv run strips env vars).
    # Mirrors the approach in _platform._find_vscode_ipc_hook() for cross-platform support.
    def _ppid(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return int(r.stdout.strip())
        except Exception:
            pass
        return -1

    pid = os.getpid()
    for _ in range(20):
        pid = _ppid(pid)
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
    if os.environ.get("TERM_PROGRAM") == "tmux":
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
                                    return entry[len(b"ARRAYVIEW_WINDOW_ID="):].decode()
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
                                return token[len("ARRAYVIEW_WINDOW_ID="):]
                    except Exception:
                        pass
        except Exception:
            pass

    return None


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


def _open_direct_via_signal_file(
    filepath: str,
    title: str | None = None,
    extra_args: list[str] | None = None,
) -> bool:
    """Write a direct-mode signal file for the VS Code extension.

    Instead of opening a URL in an iframe, the extension spawns a Python
    subprocess (``python -m arrayview --mode stdio <filepath>``) and hosts
    the viewer HTML directly in a webview panel with postMessage transport.
    No ports, no WebSocket, no authentication — just IPC.

    *extra_args* are forwarded verbatim to the subprocess command line so
    that new CLI flags (``--vectorfield``, ``--overlay``, ``--rgb``, …)
    work without touching the extension.
    """
    import sys as _sys

    payload: dict = {
        "action": "open-preview",
        "mode": "direct",
        "filepath": os.path.abspath(filepath),
        "pythonPath": _sys.executable,
        "maxAgeMs": _VSCODE_SIGNAL_MAX_AGE_MS,
    }
    if title:
        payload["title"] = title
    if extra_args:
        payload["extraArgs"] = extra_args
    return _write_vscode_signal(payload, skip_compat=True)


def _open_direct_via_shm(
    data: "np.ndarray",
    name: str = "array",
    title: str | None = None,
) -> bool:
    """Write a direct-mode signal file with shared memory parameters.

    Places the array in POSIX shared memory so the extension-spawned subprocess
    can read it without any disk I/O.
    """
    import atexit
    import sys as _sys
    from multiprocessing.shared_memory import SharedMemory

    import numpy as np

    arr = np.ascontiguousarray(data)
    shm = SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr

    # The subprocess will unlink the SHM after reading it.  Unregister from
    # Python's resource tracker so it doesn't warn about "leaked" SHM at exit.
    from multiprocessing import resource_tracker
    try:
        resource_tracker.unregister(f"/{shm.name}", "shared_memory")
    except Exception:
        pass

    # Keep the shm alive until the subprocess reads it (or this process exits).
    _ACTIVE_SHM.append(shm)

    def _cleanup():
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)

    payload: dict = {
        "action": "open-preview",
        "mode": "direct",
        "shm": {
            "name": shm.name,
            "shape": ",".join(str(int(s)) for s in arr.shape),
            "dtype": str(arr.dtype),
        },
        "arrayName": name,
        "pythonPath": _sys.executable,
        "maxAgeMs": _VSCODE_SIGNAL_MAX_AGE_MS,
    }
    if title:
        payload["title"] = title
    return _write_vscode_signal(payload, skip_compat=True)


# Shared memory blocks kept alive until process exit or subprocess reads them.
_ACTIVE_SHM: list = []


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


def _write_vscode_signal(payload: dict, delay: float = 0.0, skip_compat: bool = False) -> bool:
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

                # Guard against stale ARRAYVIEW_WINDOW_ID: when the extension
                # host restarts (e.g. tunnel reconnection), the old process may
                # still be alive but is no longer connected to a VS Code client.
                # Detect this by looking for a newer registration from the same
                # server (same first ppid — the tunnel/server parent process).
                # Only needed for PID-based IDs (fallbackId=True); hookTag IDs
                # are unique per IPC socket and never go stale across restarts.
                #
                # Skip on macOS: all extension hosts are direct children of the
                # same main Electron process (ppids[0] == Electron PID for every
                # window), so the "same ppids[0]" heuristic matches ALL concurrent
                # windows and incorrectly redirects signals to the wrong window.
                # The stable window ID feature (v0.14.0+) already handles restart
                # continuity on macOS via EnvironmentVariableCollection, making
                # this stale-redirect unnecessary there.
                if uses_pid and sys.platform != "darwin":
                    _env_ts = _reg_data.get("ts", 0)
                    _env_ppids = _reg_data.get("ppids", [])
                    try:
                        for _fname in os.listdir(signal_dir):
                            if not (_fname.startswith("window-") and _fname.endswith(".json")):
                                continue
                            _other_wid = _fname[7:-5]
                            if _other_wid == env_wid:
                                continue
                            with open(os.path.join(signal_dir, _fname)) as _f:
                                _other = json.load(_f)
                            _other_ts = _other.get("ts", 0)
                            _other_ppids = _other.get("ppids", [])
                            if (
                                _other_ts > _env_ts
                                and len(_env_ppids) >= 1
                                and len(_other_ppids) >= 1
                                and _env_ppids[0] == _other_ppids[0]
                            ):
                                _vprint(
                                    f"[ArrayView] signal: ARRAYVIEW_WINDOW_ID={env_wid} is stale "
                                    f"(newer registration {_other_wid} found), redirecting",
                                    flush=True,
                                )
                                env_wid = _other_wid
                                _reg_data = _other
                                uses_pid = _other.get("fallbackId", False)
                                _env_ts = _other_ts
                    except Exception:
                        pass

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
                    _wfiles = []
                    for _fn in _all_windows:
                        _wid = _fn[7:-5]
                        _wfiles.append(
                            f"open-request-pid-{_wid}.json"
                            if _wid.isdigit()
                            else f"open-request-ipc-{_wid}.json"
                        )
                    data["broadcast"] = True
                    filenames = tuple(_wfiles)
                    targeted_via_env = True
                    _vprint(
                        f"[ArrayView] signal: {len(_all_windows)} windows registered, "
                        f"broadcasting with focus guard",
                        flush=True,
                    )
                # else: 0 windows — fall through to subsequent targeting

        if targeted_via_env:
            pass  # filenames already set — skip to write logic below
        elif (ipc_hook := _find_vscode_ipc_hook()) and not _is_vscode_remote():
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
                            data["broadcast"] = True
                        filenames = (
                            tuple(_wfiles) if _wfiles else (_VSCODE_SIGNAL_FILENAME,)
                        )
                    else:
                        filenames = (
                            _VSCODE_SIGNAL_FILENAME,
                            *_VSCODE_COMPAT_SIGNAL_FILENAMES,
                        )
        elif not _is_vscode_remote():
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
                        # Mark as broadcast so extensions check focus before opening
                        data["broadcast"] = True
                    filenames = (
                        tuple(window_files)
                        if window_files
                        else (_VSCODE_SIGNAL_FILENAME,)
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

                    # Same stale-env guard as the primary check above.
                    _env_ts = _reg_data.get("ts", 0)
                    _env_ppids = _reg_data.get("ppids", [])
                    try:
                        for _fname in os.listdir(signal_dir):
                            if not (_fname.startswith("window-") and _fname.endswith(".json")):
                                continue
                            _other_wid = _fname[7:-5]
                            if _other_wid == env_wid:
                                continue
                            with open(os.path.join(signal_dir, _fname)) as _f:
                                _other = json.load(_f)
                            _other_ts = _other.get("ts", 0)
                            _other_ppids = _other.get("ppids", [])
                            if (
                                _other_ts > _env_ts
                                and len(_env_ppids) >= 1
                                and len(_other_ppids) >= 1
                                and _env_ppids[0] == _other_ppids[0]
                            ):
                                _vprint(
                                    f"[ArrayView] signal: remote env {env_wid} is stale "
                                    f"(newer {_other_wid}), redirecting",
                                    flush=True,
                                )
                                env_wid = _other_wid
                                _reg_data = _other
                                _uses_pid = _other.get("fallbackId", False)
                                _env_ts = _other_ts
                    except Exception:
                        pass

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
                        _wfiles_r = []
                        for _fn in _all_windows_r:
                            _wid = _fn[7:-5]
                            _wfiles_r.append(
                                f"open-request-pid-{_wid}.json"
                                if _wid.isdigit()
                                else f"open-request-ipc-{_wid}.json"
                            )
                        data["broadcast"] = True
                        filenames = tuple(_wfiles_r)
                        # env_wid is still the original stale ID (truthy), so the
                        # `if not env_wid:` fallback below won't trigger; no clobber needed.
                        _vprint(
                            f"[ArrayView] signal: remote {len(_all_windows_r)} windows, "
                            f"broadcasting with focus guard",
                            flush=True,
                        )
                    else:
                        env_wid = None  # 0 windows: keep existing fallback

            if not env_wid:
                # Fallback: PID ancestry (may misfire with multiple windows)
                _vprint(f"[ArrayView] signal: trying PID ancestry fallback", flush=True)
                window_id = _find_current_vscode_window_id()
                if window_id:
                    _vprint(f"[ArrayView] signal: PID ancestry matched window={window_id}", flush=True)
                    filenames = (
                        f"open-request-pid-{window_id}.json"
                        if window_id.isdigit()
                        else f"open-request-ipc-{window_id}.json",
                    )
                else:
                    _vprint(f"[ArrayView] signal: no window match, using shared fallback", flush=True)
                    filenames = (
                        (_VSCODE_SIGNAL_FILENAME,)
                        if skip_compat
                        else (_VSCODE_SIGNAL_FILENAME, *_VSCODE_COMPAT_SIGNAL_FILENAMES)
                    )

        _vprint(f"[ArrayView] signal: writing to {[f for f in filenames]} broadcast={data.get('broadcast', False)}", flush=True)
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


def _open_browser(
    url: str,
    blocking: bool = False,
    force_vscode: bool = False,
    title: str | None = None,
    filepath: str | None = None,
) -> None:
    """Open *url* locally, or configure VS Code remote auto-preview behavior.

    *title* is shown as the VS Code tab label (e.g. "ArrayView: sample.npy").
    Passed through to _open_via_signal_file → extension createWebviewPanel.

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
            ext_ok = _ensure_vscode_extension()
            if ext_ok and _VSCODE_EXT_FRESH_INSTALL:
                time.sleep(1.5)

            if filepath:
                # Direct webview mode: no ports, no WebSocket, no auth needed.
                # The extension spawns a Python subprocess and bridges via
                # postMessage — completely bypasses the port-forwarding issue.
                _open_direct_via_signal_file(filepath, title=title)
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
                        f"  If the Simple Browser tab shows an auth page, make the port Public then reload the tab.",
                        flush=True,
                    )
                _open_via_signal_file(url, title=title)
                _schedule_remote_open_retries(url, interval=10.0, count=2)
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
            _open_via_signal_file(url, title=title)
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
        print(f"\n  \033[1;36m→ {url}\033[0m\n", flush=True)

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()
