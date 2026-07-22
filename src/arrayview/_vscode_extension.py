"""VS Code extension installation and port management."""

from __future__ import annotations

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
_VSCODE_EXT_VERSION = "0.14.70"  # current bundled extension version
_VSCODE_CONFIGURED_PORTS: set[int] = set()

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
    import hashlib

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
        start_new_session=os.name != "nt",
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
    import hashlib

    vsix_hash = hashlib.md5(open(vsix_path, "rb").read()).hexdigest()
    for base in bases:
        ext_dir = os.path.join(base, f"arrayview.arrayview-opener-{version}")
        if os.path.isdir(ext_dir):
            with open(os.path.join(ext_dir, ".vsix_hash"), "w") as f:
                f.write(vsix_hash)


def _ensure_vscode_extension(*, is_remote: bool | None = None) -> bool:
    """Verify or install the bundled opener for the current VS Code window.

    The extension bridges local VS Code terminals to a webview panel tab
    and, in remote/tunnel sessions, can actively invoke VS Code's forwarded-port
    commands to promote the port to public preview.

    Remote installs use the exact current server CLI and must observe activation
    in the exact current window before launch can continue. Tunnel and Remote-SSH
    processes can share one extension registry, so remote cleanup never removes
    older versions and post-processing touches only the identified active base.

    The authoritative version is read from the bundled VSIX — no hardcoded
    version constant needed.
    """
    global _VSCODE_EXT_INSTALLED, _VSCODE_EXT_FRESH_INSTALL
    global _VSCODE_EXT_RELOAD_REQUIRED, _VSCODE_EXT_INSTALL_FAILED
    _VSCODE_EXT_FRESH_INSTALL = False
    _VSCODE_EXT_RELOAD_REQUIRED = False
    _VSCODE_EXT_INSTALL_FAILED = False

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
    active_marker = (
        _extension_registration_marker(active_registration)
        if active_registration is not None
        and active_registration.get("extensionVersion") == active_version
        else None
    )
    if _extension_on_disk(ext_version, vsix_path, remote=is_remote):
        active_matches = (
            active_version == ext_version
            if is_remote
            else active_version in (None, ext_version)
        )
        if not active_matches:
            _VSCODE_EXT_RELOAD_REQUIRED = True
            _vprint(
                f"[ArrayView] opener v{ext_version} is installed, but this VS Code "
                "window is still running an older extension host; reload this window",
                flush=True,
            )
            return False
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
            _VSCODE_EXT_RELOAD_REQUIRED = True
            _vprint(
                f"[ArrayView] newer opener v{newer_version} is installed, but this "
                "VS Code window is still running an older extension host; reload this window",
                flush=True,
            )
            return False
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

    env = dict(os.environ)
    ipc = _find_vscode_ipc_hook()
    if ipc:
        env["VSCODE_IPC_HOOK_CLI"] = ipc

    install_snapshot = _extension_base_snapshot() if is_remote else {}
    try:
        r = _run_extension_installer(
            [code, "--install-extension", vsix_path, "--force"], env
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
        print(f"[ArrayView] extension install failed: {combined.strip()!r}", flush=True)
    except Exception as exc:
        if active_version not in (None, ext_version):
            _VSCODE_EXT_RELOAD_REQUIRED = True
        if is_remote:
            _VSCODE_EXT_INSTALL_FAILED = True
        print(f"[ArrayView] extension install error: {exc}", flush=True)
    return False

# ---------------------------------------------------------------------------
# Port settings
# ---------------------------------------------------------------------------


def _configure_vscode_port_preview(
    port: int,
    *,
    in_vscode: bool | None = None,
    is_remote: bool | None = None,
) -> bool:
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
        desired = {
            "protocol": "http",
            "label": "ArrayView",
            "onAutoForward": "silent",
            "privacy": "public",
        }
        current = attrs.get(str(port))
        updated = {**current, **desired} if isinstance(current, dict) else desired
        if current == updated:
            return
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

        if is_remote:
            home = os.path.expanduser("~")
            targets: list[str] = []
            # `code tunnel` keeps its data under ~/.vscode/data/ and its
            # extensions under ~/.vscode/extensions/.  SSH-Remote uses
            # ~/.vscode-server/data/.  The `code tunnel` CLI server is rooted
            # at ~/.vscode/cli/.  Write Machine/User settings to all three so
            # the port auto-forwards as public regardless of transport.
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
                # Fallback: write to the most common paths even if root
                # directories don't exist yet.
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
