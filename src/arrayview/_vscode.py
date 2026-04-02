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

    # Fast path: correct version already installed — no reinstall needed.
    # Reinstalling with --force triggers an extension-host reload, which
    # creates a ~10-15s gap during which the signal file can be missed.
    _remove_old_extension_versions(ext_version)
    if _extension_on_disk(ext_version):
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

    On macOS, VS Code spawns per-window renderer processes. The extension host
    and the terminal's PTY host are both children of the same renderer, giving
    them a common ancestor that distinguishes them from other windows. We find
    the window whose extension host shares the closest (deepest) common ancestor
    with our own process tree.

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

    # Fallback: no ppids in registrations (older extension version) or no match.
    # Use process start time proximity as a heuristic.
    from arrayview._platform import _in_vscode_terminal

    if _in_vscode_terminal():
        try:
            terminal_ppid = os.getppid()
            r2 = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(terminal_ppid)],
                capture_output=True, text=True, timeout=1,
            )
            term_start = r2.stdout.strip()
            closest_id: str | None = None
            closest_delta = float("inf")
            for window_id, ext_pid, _ in windows:
                if not ext_pid:
                    continue
                try:
                    r = subprocess.run(
                        ["ps", "-o", "lstart=", "-p", str(ext_pid)],
                        capture_output=True, text=True, timeout=1,
                    )
                    ext_start = r.stdout.strip()
                    if ext_start and term_start and ext_start <= term_start:
                        delta = abs(hash(ext_start) - hash(term_start))
                        if closest_id is None or delta < closest_delta:
                            closest_id = window_id
                            closest_delta = delta
                except Exception:
                    continue
            if closest_id:
                return closest_id
        except Exception:
            pass

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

    # Walk ancestors looking for ARRAYVIEW_WINDOW_ID in /proc/<pid>/environ
    def _ppid(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return -1

    pid = os.getpid()
    for _ in range(12):
        pid = _ppid(pid)
        if pid <= 1:
            break
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                for entry in fh.read().split(b"\0"):
                    if entry.startswith(b"ARRAYVIEW_WINDOW_ID="):
                        return entry[len(b"ARRAYVIEW_WINDOW_ID="):].decode()
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
    filepath: str, title: str | None = None
) -> bool:
    """Write a direct-mode signal file for the VS Code extension.

    Instead of opening a URL in an iframe, the extension spawns a Python
    subprocess (``python -m arrayview --mode stdio <filepath>``) and hosts
    the viewer HTML directly in a webview panel with postMessage transport.
    No ports, no WebSocket, no authentication — just IPC.
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

    Signal-file targeting strategy:

    LOCAL VS Code (not remote):
      When VSCODE_IPC_HOOK_CLI can be found (directly or via parent-process
      walk), Python computes hookTag = SHA-256(ipc_hook)[:16] and writes *only*
      to the per-window targeted file
      ``~/.arrayview/open-request-ipc-<hookTag>.json``.  The extension in the
      correct VS Code window checks that same file and claims it exclusively.
      This prevents multi-window races.

    REMOTE / TUNNEL VS Code:
      The terminal and extension host have *different* IPC hooks (different
      process trees), so hookTag matching can't work directly.  Instead,
      Python uses PID ancestry matching (``_find_current_vscode_window_id``)
      to identify which extension host's window spawned this terminal, then
      writes to that extension's targeted signal file.  Falls back to the
      shared signal file only when no window match is found.

    Falls back to the shared signal file when ipc_hook is not found at all
    (e.g. non-VS Code environment or fully-stripped env).
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

        ipc_hook = _find_vscode_ipc_hook()
        if ipc_hook and not _is_vscode_remote():
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
                    _vprint(f"[ArrayView] signal: env window match → ipc-{env_wid}", flush=True)
                    filenames = (f"open-request-ipc-{env_wid}.json",)
                else:
                    _vprint(f"[ArrayView] signal: env window {env_wid} not registered, falling back", flush=True)
                    env_wid = None

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
            try:
                port_hint = int(url.split(":")[2].split("/")[0].split("?")[0])
            except Exception:
                port_hint = parsed_port
            print(
                f"[ArrayView] SSH session — automatic relay to VS Code failed.\n"
                f"\n"
                f"  If the VS Code arrayview extension is running on the machine you\n"
                f"  SSHed from, the array normally opens automatically. If it didn't,\n"
                f"  the extension may not be installed or the network may block the\n"
                f"  relay port (default 17789).\n"
                f"\n"
                f"  Manual alternatives:\n"
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
