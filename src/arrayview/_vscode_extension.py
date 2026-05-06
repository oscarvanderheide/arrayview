"""VS Code extension installation and port management."""

from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from importlib.resources import files as _pkg_files

from arrayview._session import _vprint
from arrayview._platform import _find_code_cli, _find_vscode_ipc_hook, _in_vscode_terminal, _is_vscode_remote

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
_VSCODE_EXT_VERSION = "0.14.4"  # current bundled extension version

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

    The extension bridges local VS Code terminals to a webview panel tab
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
