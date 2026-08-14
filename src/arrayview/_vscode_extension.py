"""VS Code extension installation and port management."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
import zipfile
from importlib.resources import files as _pkg_files

from arrayview._session import _vprint
from arrayview._platform import (
    _exact_vscode_window_registration,
    _find_code_cli,
    _find_vscode_ipc_hook,
    _in_vscode_terminal,
    _is_vscode_remote,
    _process_is_alive,
)

# ---------------------------------------------------------------------------
# VS Code .app bundle detection (macOS)
# ---------------------------------------------------------------------------


def _vscode_app_bundle() -> str | None:
    """Return the path to the VS Code .app bundle on macOS, derived from the code CLI."""
    code = _find_code_cli(is_remote=False)
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
_VSCODE_EXT_RELOAD_REQUIRED = False  # installed files are newer than the live host
_VSCODE_EXT_INSTALL_FAILED = False  # automatic install could not complete safely
_VSCODE_EXT_NO_LIVE_WINDOW = False  # no live host claims this terminal's window id
_VSCODE_EXT_VERSION = "0.15.48"  # current bundled extension version
_VSCODE_CONFIGURED_PORTS: set[int] = set()
# Version-skew notices already printed, so a long-lived process (Jupyter, a
# script opening several arrays) states the mismatch once instead of per call.
_VSCODE_EXT_SKEW_REPORTED: set[tuple] = set()
_VSCODE_EXT_INSTALL_GUARD_PREFIX = "vscode-extension-install-failure-"
_VSCODE_EXT_INSTALL_NO_HOST_COOLDOWN_SECONDS = 300.0


def _extension_install_guard_key(
    version: str, code: str, registration_marker: tuple | None
) -> dict:
    """Return the exact installer target state protected by a failure guard."""
    return {
        "version": version,
        "codeCli": os.path.realpath(code),
        "registrationMarker": (
            list(registration_marker) if registration_marker is not None else None
        ),
    }


def _extension_install_guard_path(key: dict) -> str:
    encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    return os.path.join(
        os.path.expanduser("~/.arrayview"),
        f"{_VSCODE_EXT_INSTALL_GUARD_PREFIX}{digest}.json",
    )


def _write_extension_install_failure_guard(
    version: str,
    code: str,
    registration_marker: tuple | None,
    message: str,
) -> bool:
    """Persist one failed install transaction so new processes do not repeat it."""
    key = _extension_install_guard_key(version, code, registration_marker)
    path = _extension_install_guard_path(key)
    payload = {**key, "failedAt": time.time(), "message": message}
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        _vprint(
            f"[ArrayView] could not persist VS Code installer failure guard: {exc}",
            flush=True,
        )
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _clear_extension_install_failure_guards(version: str, code: str) -> None:
    """Clear prior failures for a profile after that profile installs successfully."""
    signal_dir = os.path.expanduser("~/.arrayview")
    wanted_cli = os.path.realpath(code)
    try:
        filenames = os.listdir(signal_dir)
    except OSError:
        return
    for filename in filenames:
        if not (
            filename.startswith(_VSCODE_EXT_INSTALL_GUARD_PREFIX)
            and filename.endswith(".json")
        ):
            continue
        path = os.path.join(signal_dir, filename)
        try:
            with open(path) as handle:
                payload = json.load(handle)
            if (
                payload.get("version") == version
                and payload.get("codeCli") == wanted_cli
            ):
                os.unlink(path)
        except (OSError, ValueError, TypeError, AttributeError):
            continue


def _extension_install_failure_guard(
    version: str,
    code: str,
    registration_marker: tuple | None,
    *,
    now: float | None = None,
) -> dict | None:
    """Return an applicable durable guard for this exact live installer target.

    A live registration marker makes the guard valid until that extension host
    changes. Without a live marker, allow another attempt after a bounded
    cooldown because there is no reconnect evidence that can invalidate it.
    """
    key = _extension_install_guard_key(version, code, registration_marker)
    path = _extension_install_guard_path(key)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or any(
        payload.get(field) != value for field, value in key.items()
    ):
        return None
    if registration_marker is not None:
        return payload
    failed_at = payload.get("failedAt")
    if not isinstance(failed_at, (int, float)):
        return None
    current = time.time() if now is None else now
    if current - failed_at < _VSCODE_EXT_INSTALL_NO_HOST_COOLDOWN_SECONDS:
        return payload
    try:
        os.unlink(path)
    except OSError:
        pass
    return None

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


def _patch_vscode_extension_metadata(
    version: str, *, bases: tuple[str, ...] | None = None
) -> None:
    """Remove broken targetPlatform metadata written by VS Code for local VSIX installs."""
    if bases is None:
        bases = (
            os.path.expanduser("~/.vscode-server/extensions"),
            os.path.expanduser("~/.vscode/extensions"),
        )
    for base_dir in bases:
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


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """Return a numeric version tuple, or ``None`` for unknown formats."""
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, TypeError, ValueError):
        return None


def _extension_bases(*, remote: bool | None = None) -> tuple[str, ...]:
    """Return extension roots for the active VS Code host."""
    local = os.path.expanduser("~/.vscode/extensions")
    remote_base = os.path.expanduser("~/.vscode-server/extensions")
    if remote is True:
        if os.path.isdir(os.path.expanduser("~/.vscode-server")):
            return (remote_base,)
        # `code tunnel` uses ~/.vscode/extensions on the remote host, while
        # Remote-SSH uses ~/.vscode-server/extensions.
        return (local,)
    if remote is False:
        return (local,)
    return (local, remote_base)


def _remove_old_extension_versions(
    current_version: str, *, remote: bool | None = None
) -> None:
    """Delete older extension directories without ever deleting a newer build.

    When multiple versions of arrayview-opener are installed side-by-side,
    VS Code may load an older version instead of the latest.  Removing stale
    directories ensures the correct version is picked up on the next reload.
    Cleanup is scoped to the active local or remote extension host.
    """
    import shutil

    current_key = _version_tuple(current_version)
    for ext_base in _extension_bases(remote=remote):
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
            version_key = _version_tuple(version_str)
            if current_key is None or version_key is None or version_key > current_key:
                continue
            old_dir = os.path.join(ext_base, entry)
            try:
                shutil.rmtree(old_dir)
                _vprint(f"[ArrayView] removed old extension: {entry}", flush=True)
            except Exception as exc:
                _vprint(f"[ArrayView] could not remove {entry}: {exc}", flush=True)


def _extension_on_disk(
    version: str,
    vsix_path: str | None = None,
    *,
    remote: bool | None = None,
) -> bool:
    """Return True if the extension directory for *version* exists on disk.

    When *vsix_path* is given, also verifies that the installed extension
    matches the bundled VSIX.  The hash written by ArrayView is only a fast
    path: VS Code may install the extension without that private marker, so a
    missing or stale marker falls back to comparing the packaged files before
    deciding that a reinstall is necessary.
    """
    for base in _extension_bases(remote=remote):
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
        if _installed_extension_matches_vsix(ext_dir, vsix_path):
            # The marker is an ArrayView optimization, not part of the VSIX.
            # Backfill it after a verified VS Code install so ordinary launches
            # do not force a redundant --force install and window reload.
            try:
                with open(hash_file, "w") as f:
                    f.write(vsix_hash)
            except OSError as exc:
                _vprint(
                    f"[ArrayView] could not cache verified VSIX hash at "
                    f"{hash_file}: {exc}",
                    flush=True,
                )
            return True
        _vprint(
            f"[ArrayView] VSIX content changed (installed={installed_hash}, bundled={vsix_hash}) — reinstalling",
            flush=True,
        )
        return False
    return False


def _installed_extension_matches_vsix(ext_dir: str, vsix_path: str) -> bool:
    """Return whether all files shipped in *vsix_path* match *ext_dir*.

    VS Code injects a top-level ``__metadata`` object into ``package.json``
    and can rewrite its formatting during installation.  That field is host
    bookkeeping rather than bundled extension content, so package manifests
    are compared structurally after removing it.  Every other shipped file is
    compared byte-for-byte.  Extra host files such as ``.vsix_hash`` do not
    affect the result.
    """

    try:
        with zipfile.ZipFile(vsix_path) as zf:
            packaged_files = [
                info
                for info in zf.infolist()
                if info.filename.startswith("extension/") and not info.is_dir()
            ]
            if not packaged_files:
                return False
            for info in packaged_files:
                relative = info.filename.removeprefix("extension/")
                parts = relative.split("/")
                if not relative or any(part in ("", ".", "..") for part in parts):
                    return False
                installed_path = os.path.join(ext_dir, *parts)
                if not os.path.isfile(installed_path):
                    return False
                packaged = zf.read(info)
                with open(installed_path, "rb") as f:
                    installed = f.read()
                if relative == "package.json":
                    packaged_json = json.loads(packaged)
                    installed_json = json.loads(installed)
                    if not isinstance(packaged_json, dict) or not isinstance(
                        installed_json, dict
                    ):
                        return False
                    packaged_json.pop("__metadata", None)
                    installed_json.pop("__metadata", None)
                    if installed_json != packaged_json:
                        return False
                elif installed != packaged:
                    return False
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile):
        return False
    return True


def _newer_extension_on_disk(version: str, *, remote: bool) -> str | None:
    """Return the newest installed opener newer than *version*, if any."""
    wanted = _version_tuple(version)
    if wanted is None:
        return None
    prefix = "arrayview.arrayview-opener-"
    newer: list[tuple[tuple[int, ...], str]] = []
    for base in _extension_bases(remote=remote):
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith(prefix):
                continue
            candidate = entry[len(prefix) :]
            key = _version_tuple(candidate)
            if key is not None and key > wanted:
                newer.append((key, candidate))
    return max(newer)[1] if newer else None


def _live_window_registrations() -> list[dict]:
    """Return every live opener registration under ``~/.arrayview``.

    Each open VS Code window writes one of these files and then keeps running
    whatever opener build it loaded when it activated.  Reading them all is the
    only way a terminal can see which versions its sibling windows are on.
    """
    signal_dir = os.path.expanduser("~/.arrayview")
    registrations: list[dict] = []
    try:
        filenames = sorted(os.listdir(signal_dir))
    except OSError:
        return registrations
    for filename in filenames:
        if not filename.startswith("window-") or not filename.endswith(".json"):
            continue
        try:
            with open(
                os.path.join(signal_dir, filename), encoding="utf-8"
            ) as handle:
                registration = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(registration, dict):
            continue
        if not _process_is_alive(registration.get("pid")):
            continue
        registrations.append(registration)
    return registrations


def _window_version_summary(registration: dict) -> dict:
    """Return the display-ready identity of one live window."""
    version = registration.get("extensionVersion")
    window_id = registration.get("windowId")
    return {
        "opener_version": version if isinstance(version, str) and version else None,
        "window_id": window_id if isinstance(window_id, str) else None,
        "pid": registration.get("pid"),
        "host": registration.get("remoteName") or "local",
    }


def _describe_window(summary: dict) -> str:
    """Render one window as ``v0.15.16  (window 0761ea2d, pid 644152, tunnel)``."""
    version = summary.get("opener_version")
    label = f"v{version}" if version else "unknown version (pre-0.15 build)"
    window_id = summary.get("window_id") or "?"
    return (
        f"{label}  (window {window_id[:8]}, pid {summary.get('pid') or '?'}, "
        f"{summary.get('host')})"
    )


def _opener_version_report(bundled_version: str | None = None) -> dict:
    """Summarise every version that decides which opener a launch will use.

    Three things can disagree and only one of them is visible from the shell
    prompt: the ``arrayview`` on this terminal's PATH, the opener that copy of
    arrayview ships, and the opener this VS Code window actually has loaded.
    """
    from arrayview import __version__ as package_version

    if bundled_version is None:
        vsix_path = str(_pkg_files("arrayview").joinpath("arrayview-opener.vsix"))
        bundled_version = (
            _bundled_vscode_vsix_version(vsix_path)
            if os.path.isfile(vsix_path)
            else None
        ) or _VSCODE_EXT_VERSION

    active = _active_extension_registration()
    this_window = _window_version_summary(active) if active is not None else None
    this_id = this_window.get("window_id") if this_window else None
    others = [
        _window_version_summary(registration)
        for registration in _live_window_registrations()
        if registration.get("windowId") != this_id
    ]

    live_version = this_window.get("opener_version") if this_window else None
    if this_window is None:
        state = "no_live_window"
    elif live_version is None:
        state = "window_unversioned"
    elif live_version == bundled_version:
        state = "match"
    else:
        bundled_key = _version_tuple(bundled_version)
        live_key = _version_tuple(live_version)
        if bundled_key is not None and live_key is not None and live_key > bundled_key:
            state = "window_newer"
        else:
            state = "window_older"

    return {
        "package_version": package_version,
        "package_path": os.path.dirname(os.path.abspath(__file__)),
        "bundled_opener": bundled_version,
        "this_window": this_window,
        "other_windows": others,
        "state": state,
    }


def _version_skew_lines(report: dict) -> list[str]:
    """Return the terminal notice for a report, or ``[]`` when versions agree."""
    state = report["state"]
    if state in {"match", "no_live_window"}:
        return []
    this_window = report["this_window"] or {}
    live = this_window.get("opener_version")
    live_label = f"v{live}" if live else "an unversioned build"
    lines = [
        "[ArrayView] This VS Code window is not running the opener this "
        "arrayview ships.",
        f"  this terminal:  arrayview {report['package_version']} "
        f"ships opener v{report['bundled_opener']}",
        f"                  {report['package_path']}",
        f"  this window:    opener {live_label} "
        f"(fixed when the window last reloaded)",
    ]
    if state == "window_newer":
        lines.append(
            "  A newer arrayview in another window installed it. The newer "
            "opener is kept and used."
        )
    else:
        lines.append(
            '  Reload this window (Ctrl+Shift+P → "Developer: Reload Window") '
            f"to run v{report['bundled_opener']}."
        )
    return lines


def _report_opener_version_skew(bundled_version: str | None = None) -> bool:
    """Print an up-front notice when this window's opener is not the bundled one.

    Announced before the launch is attempted rather than only in the failure
    message afterwards: with several windows open, each pinned to whichever
    opener it loaded at activation, "which build am I about to run" is not
    answerable from the prompt.
    """
    global _VSCODE_EXT_SKEW_REPORTED
    report = _opener_version_report(bundled_version)
    lines = _version_skew_lines(report)
    if not lines:
        return False
    this_window = report["this_window"] or {}
    fingerprint = (
        report["bundled_opener"],
        this_window.get("opener_version"),
        this_window.get("window_id"),
    )
    if fingerprint in _VSCODE_EXT_SKEW_REPORTED:
        return False
    _VSCODE_EXT_SKEW_REPORTED.add(fingerprint)
    print("\n".join(lines), flush=True)
    return True


def _active_extension_registration() -> dict | None:
    """Return the live opener registration for this terminal's exact window."""
    ipc = _find_vscode_ipc_hook()
    exact = _exact_vscode_window_registration(ipc)
    return exact[1] if exact is not None else None


def _active_extension_version() -> str | None:
    """Return the opener version advertised by this terminal's live host."""
    registration = _active_extension_registration()
    if registration is None:
        return None
    value = registration.get("extensionVersion")
    # An existing registration without a version belongs to a legacy host.
    return value if isinstance(value, str) else ""


def _extension_registration_marker(registration: dict | None) -> tuple | None:
    if registration is None:
        return None
    return (
        registration.get("pid"),
        registration.get("extensionInstanceId"),
        registration.get("ts"),
    )


def _wait_for_active_extension_version(
    version: str,
    timeout: float = 15.0,
    *,
    previous_marker: tuple | None = None,
) -> bool:
    """Wait for an updated extension host registration after installation."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        registration = _active_extension_registration()
        marker = _extension_registration_marker(registration)
        if (
            registration is not None
            and registration.get("extensionVersion") == version
            and (previous_marker is None or marker != previous_marker)
        ):
            return True
        time.sleep(0.1)
    registration = _active_extension_registration()
    marker = _extension_registration_marker(registration)
    return bool(
        registration is not None
        and registration.get("extensionVersion") == version
        and (previous_marker is None or marker != previous_marker)
    )


def _run_extension_installer(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run VS Code's installer without leaving its server-cli child behind."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        # Same intent on both platforms: the installer gets its own group, so
        # neither its shutdown nor a console Ctrl-C crosses between us and it.
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _extension_base_snapshot() -> dict[str, tuple[int, int, tuple[str, ...]]]:
    """Return registry/directory mtimes used to identify one install target."""
    snapshot: dict[str, tuple[int, int, tuple[str, ...]]] = {}
    for base in _extension_bases(remote=None):
        registry = os.path.join(base, "extensions.json")
        try:
            registry_mtime = os.stat(registry).st_mtime_ns
        except OSError:
            registry_mtime = -1
        try:
            base_mtime = os.stat(base).st_mtime_ns
        except OSError:
            base_mtime = -1
        try:
            entries = tuple(sorted(os.listdir(base)))
        except OSError:
            entries = ()
        snapshot[base] = (registry_mtime, base_mtime, entries)
    return snapshot


def _remote_install_base(
    *,
    active_version: str | None,
    version: str,
    before: dict[str, tuple[int, int, tuple[str, ...]]],
) -> str | None:
    """Identify the single remote extension root changed by the active CLI."""
    bases = _extension_bases(remote=None)
    after = _extension_base_snapshot()
    if active_version:
        active_bases = [
            base
            for base in bases
            if os.path.isdir(
                os.path.join(base, f"arrayview.arrayview-opener-{active_version}")
            )
        ]
        if len(active_bases) == 1:
            active_base = active_bases[0]
            desired = os.path.join(
                active_base, f"arrayview.arrayview-opener-{version}"
            )
            return (
                active_base
                if os.path.isdir(desired)
                and after.get(active_base) != before.get(active_base)
                else None
            )

    changed = [base for base in bases if after.get(base) != before.get(base)]
    if len(changed) == 1:
        return changed[0]

    installed = [
        base
        for base in bases
        if os.path.isdir(os.path.join(base, f"arrayview.arrayview-opener-{version}"))
    ]
    return installed[0] if len(installed) == 1 else None


def _write_vscode_extension_hash(
    version: str, vsix_path: str, *, bases: tuple[str, ...]
) -> None:
    """Record the bundled content hash only in the selected installation root."""
    vsix_hash = hashlib.md5(open(vsix_path, "rb").read()).hexdigest()
    for base in bases:
        ext_dir = os.path.join(base, f"arrayview.arrayview-opener-{version}")
        if os.path.isdir(ext_dir):
            with open(os.path.join(ext_dir, ".vsix_hash"), "w") as f:
                f.write(vsix_hash)


def _ensure_vscode_extension(*, is_remote: bool | None = None) -> bool:
    """Verify or install the bundled opener for the current VS Code window.

    The extension bridges local VS Code terminals to a viewer tab and remote
    sessions to a private client-side display surface.

    Remote installs use the exact current server CLI and must observe activation
    in the exact current window before launch can continue. Tunnel and Remote-SSH
    processes can share one extension registry, so remote cleanup never removes
    older versions and post-processing touches only the identified active base.

    The authoritative version is read from the bundled VSIX — no hardcoded
    version constant needed.
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    global _VSCODE_EXT_RELOAD_REQUIRED, _VSCODE_EXT_INSTALL_FAILED
    global _VSCODE_EXT_NO_LIVE_WINDOW
    _VSCODE_EXT_FRESH_INSTALL = False
    _VSCODE_EXT_RELOAD_REQUIRED = False
    _VSCODE_EXT_INSTALL_FAILED = False
    _VSCODE_EXT_NO_LIVE_WINDOW = False

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
    if is_remote is None:
        is_remote = _is_vscode_remote()
    active_version = _active_extension_version()
    active_registration = _active_extension_registration()
    # Before the slow install/verify work, not only in the failure text after
    # it: the user needs to know which opener this window will run.
    _report_opener_version_skew(ext_version)
    active_marker = (
        _extension_registration_marker(active_registration)
        if active_registration is not None
        and active_registration.get("extensionVersion") == active_version
        else None
    )
    def _reject_unmatched_host(installed_version: str) -> bool:
        """Record why the live host was rejected, so callers can advise correctly.

        A version disagreement is fixed by reloading; an unidentifiable window is
        not.  Conflating them tells the user to reload when the reload is what
        stranded their terminal in the first place.
        """
        global _VSCODE_EXT_RELOAD_REQUIRED, _VSCODE_EXT_NO_LIVE_WINDOW
        if active_version is None:
            _VSCODE_EXT_NO_LIVE_WINDOW = True
            _vprint(
                f"[ArrayView] opener v{installed_version} is installed, but no live "
                "VS Code window claims this terminal; open a new terminal in the "
                "target window",
                flush=True,
            )
            return False
        _VSCODE_EXT_RELOAD_REQUIRED = True
        _vprint(
            f"[ArrayView] opener v{installed_version} is installed, but this VS Code "
            "window is still running an older extension host; reload this window",
            flush=True,
        )
        return False

    if _extension_on_disk(ext_version, vsix_path, remote=is_remote):
        active_matches = (
            active_version == ext_version
            if is_remote
            else active_version in (None, ext_version)
        )
        if not active_matches:
            return _reject_unmatched_host(ext_version)
        if not is_remote:
            _remove_old_extension_versions(ext_version, remote=False)
        _VSCODE_EXT_INSTALLED = True
        _vprint(
            f"[ArrayView] extension v{ext_version} already installed — skipping reinstall",
            flush=True,
        )
        return True

    newer_version = _newer_extension_on_disk(ext_version, remote=is_remote)
    if newer_version is not None:
        active_matches = (
            active_version == newer_version
            if is_remote
            else active_version in (None, newer_version)
        )
        if not active_matches:
            return _reject_unmatched_host(newer_version)
        _VSCODE_EXT_INSTALLED = True
        _vprint(
            f"[ArrayView] newer extension v{newer_version} is installed — keeping it",
            flush=True,
        )
        return True

    code = _find_code_cli(is_remote=is_remote)
    if not code:
        if is_remote:
            _VSCODE_EXT_INSTALL_FAILED = True
        return False
    install_guard = _extension_install_failure_guard(
        ext_version, code, active_marker
    )
    if install_guard is not None:
        _VSCODE_EXT_INSTALL_FAILED = True
        prior_message = install_guard.get("message")
        detail = f": {prior_message}" if isinstance(prior_message, str) else ""
        if active_marker is None:
            guidance = "waiting for the bounded installer cooldown"
        else:
            guidance = "retry after this VS Code window reloads or reconnects"
        print(
            "[ArrayView] skipping a repeated VS Code opener install against the "
            f"same unchanged host ({guidance}){detail}",
            flush=True,
        )
        return False

    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    install_snapshot = _extension_base_snapshot() if is_remote else {}
    # Say so before the slow part, not after it. Installing the opener takes
    # several seconds and may end in "reload this window once"; announcing that
    # only in the final error message left the terminal looking hung, with the
    # explanation arriving after the wait instead of before it.
    print(
        f"[ArrayView] installing the VS Code opener extension (v{ext_version})… "
        "a one-time window reload may be needed afterwards",
        flush=True,
    )
    try:
        r = _run_extension_installer(
            [code, "--install-extension", vsix_path], env
        )
        combined = (r.stdout or "") + (r.stderr or "")
        install_failed = (
            "Cannot install" in combined
            or "Failed Installing Extensions" in combined
            or "extension/package.json not found inside zip" in combined
            or "Error:" in combined
        )
        if r.returncode == 0 and not install_failed:
            install_bases: tuple[str, ...]
            if is_remote:
                active_base = _remote_install_base(
                    active_version=active_version,
                    version=ext_version,
                    before=install_snapshot,
                )
                if active_base is None:
                    _VSCODE_EXT_INSTALL_FAILED = True
                    _write_extension_install_failure_guard(
                        ext_version,
                        code,
                        active_marker,
                        "the active remote extension profile could not be identified",
                    )
                    print(
                        "[ArrayView] extension installed, but its active remote "
                        "profile could not be identified safely; no other VS Code "
                        "profile was modified",
                        flush=True,
                    )
                    return False
                install_bases = (active_base,)
            else:
                install_bases = _extension_bases(remote=False)
            _patch_vscode_extension_metadata(ext_version, bases=install_bases)
            try:
                _write_vscode_extension_hash(
                    ext_version, vsix_path, bases=install_bases
                )
            except Exception:
                pass  # non-critical
            _clear_extension_install_failure_guards(ext_version, code)
            _VSCODE_EXT_INSTALLED = True
            _VSCODE_EXT_FRESH_INSTALL = True
            if not is_remote:
                _remove_old_extension_versions(ext_version, remote=False)
            else:
                # Do not write a viewer request until this exact window has
                # advertised the installed version. A bounded wait handles VS
                # Code's normal hot activation; otherwise one reload is needed.
                if _wait_for_active_extension_version(
                    ext_version, previous_marker=active_marker
                ):
                    return True
                _VSCODE_EXT_RELOAD_REQUIRED = True
                return False
            if active_version not in (None, ext_version):
                if not _wait_for_active_extension_version(ext_version):
                    _VSCODE_EXT_RELOAD_REQUIRED = True
                    _vprint(
                        "[ArrayView] VS Code installed the new opener, but this window "
                        "is still running the old extension host; reload this VS Code window",
                        flush=True,
                    )
                    return False
            return True
        if active_version not in (None, ext_version):
            _VSCODE_EXT_RELOAD_REQUIRED = True
        if is_remote:
            _VSCODE_EXT_INSTALL_FAILED = True
        failure_message = combined.strip() or f"installer exited with {r.returncode}"
        _write_extension_install_failure_guard(
            ext_version, code, active_marker, failure_message
        )
        print(f"[ArrayView] extension install failed: {failure_message!r}", flush=True)
    except Exception as exc:
        if active_version not in (None, ext_version):
            _VSCODE_EXT_RELOAD_REQUIRED = True
        if is_remote:
            _VSCODE_EXT_INSTALL_FAILED = True
        _write_extension_install_failure_guard(
            ext_version, code, active_marker, str(exc)
        )
        print(f"[ArrayView] extension install error: {exc}", flush=True)
    return False

# ---------------------------------------------------------------------------
# Port settings
# ---------------------------------------------------------------------------


def _port_has_listener(port: int) -> bool:
    """Return True when something is listening on ``port`` on loopback."""
    import socket

    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.15)
            if sock.connect_ex((addr, port)) == 0:
                return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _stale_arrayview_port_keys(attrs: dict, keep_port: int) -> list[str]:
    """Return ArrayView port entries that no longer have a live server.

    Every launch on a fresh port used to leave its entry behind for good. On a
    tunnel those entries keep telling VS Code to forward ports nothing serves,
    which competes for a limited per-tunnel forwarding budget. Only entries
    ArrayView wrote are considered, and only when their port is dead, so a
    second concurrent ArrayView server is never disturbed.
    """
    stale: list[str] = []
    for key, value in list(attrs.items()):
        if not isinstance(value, dict) or value.get("label") != "ArrayView":
            continue
        if key == str(keep_port):
            continue
        try:
            candidate = int(key)
        except (TypeError, ValueError):
            continue
        if not _port_has_listener(candidate):
            stale.append(key)
    return stale


def _configure_vscode_port_preview(
    port: int,
    *,
    in_vscode: bool | None = None,
    is_remote: bool | None = None,
    is_tunnel: bool | None = None,
) -> bool:
    """Write VS Code port settings for the arrayview server.

    VS Code tunnels use the opener's private remote proxy and must not persist
    port-forwarding settings. Remote SSH retains its existing Machine/User
    settings behavior.

    In local VS Code terminals we keep the workspace-level attribute so
    auto-forward/silent remains configured when relevant.

    Returns True on success.
    """
    legacy_arrayview_attributes = {
        "protocol": "http",
        "label": "ArrayView",
        "onAutoForward": "silent",
        "privacy": "public",
    }

    def _strip_json_comments(raw: str) -> str:
        raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
        raw = re.sub(r"(^|\s)//.*$", "", raw, flags=re.MULTILINE)
        return raw

    def _load_settings(path: str) -> dict | None:
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = f.read()
            cleaned = _strip_json_comments(raw)
            return json.loads(cleaned) if cleaned.strip() else {}
        except (json.JSONDecodeError, OSError) as exc:
            _vprint(
                f"[ArrayView] leaving unreadable VS Code settings unchanged at "
                f"{path}: {exc}",
                flush=True,
            )
            return None

    def _write_settings(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        settings = _load_settings(path)
        if settings is None:
            return
        attrs = settings.setdefault("remote.portsAttributes", {})
        desired = legacy_arrayview_attributes
        current = attrs.get(str(port))
        updated = {**current, **desired} if isinstance(current, dict) else desired
        stale = _stale_arrayview_port_keys(attrs, port)
        if current == updated and not stale:
            return
        for key in stale:
            attrs.pop(key, None)
        attrs[str(port)] = updated
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")

    try:
        if port in _VSCODE_CONFIGURED_PORTS:
            return True
        if in_vscode is None:
            in_vscode = _in_vscode_terminal()
        if is_remote is None:
            is_remote = _is_vscode_remote()
        if is_tunnel is None:
            from arrayview._platform import _in_vscode_tunnel

            is_tunnel = _in_vscode_tunnel()

        if is_tunnel:
            home = os.path.expanduser("~")
            for root in (
                os.path.join(home, ".vscode"),
                os.path.join(home, ".vscode", "cli"),
                os.path.join(home, ".vscode-server"),
            ):
                for scope in ("Machine", "User"):
                    settings_path = os.path.join(
                        root, "data", scope, "settings.json"
                    )
                    settings = _load_settings(settings_path)
                    if settings is None:
                        continue
                    attrs = settings.get("remote.portsAttributes")
                    if not isinstance(attrs, dict):
                        continue
                    changed = False
                    for key, value in tuple(attrs.items()):
                        if value == legacy_arrayview_attributes:
                            attrs.pop(key)
                            changed = True
                    if changed:
                        with open(settings_path, "w") as f:
                            json.dump(settings, f, indent=2)
                            f.write("\n")
            return True

        if is_remote:
            home = os.path.expanduser("~")
            targets: list[str] = []
            # Remote SSH installations can use either server data root.
            for root in (
                os.path.join(home, ".vscode"),
                os.path.join(home, ".vscode", "cli"),
                os.path.join(home, ".vscode-server"),
            ):
                if os.path.isdir(root):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))
            if not targets:
                # Fallback: write to the common remote-server paths.
                for root in (
                    os.path.join(home, ".vscode"),
                    os.path.join(home, ".vscode-server"),
                    os.path.join(home, ".vscode", "cli"),
                ):
                    targets.append(
                        os.path.join(root, "data", "Machine", "settings.json")
                    )
                    targets.append(os.path.join(root, "data", "User", "settings.json"))

            for settings_path in targets:
                _write_settings(settings_path)
            _VSCODE_CONFIGURED_PORTS.add(port)
            return True

        if in_vscode:
            settings_path = os.path.join(os.getcwd(), ".vscode", "settings.json")
            _write_settings(settings_path)
            _VSCODE_CONFIGURED_PORTS.add(port)
        return True
    except Exception as exc:
        _vprint(f"[ArrayView] could not write port settings: {exc}", flush=True)
        return False
