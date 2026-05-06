"""VS Code integration — facade re-exporting from submodules."""

from arrayview._vscode_extension import (  # noqa: F401
    _bundled_vscode_vsix_version,
    _configure_vscode_port_preview,
    _ensure_vscode_extension,
    _extension_on_disk,
    _patch_vscode_extension_metadata,
    _remove_old_extension_versions,
    _vscode_app_bundle,
    _VSCODE_EXT_FRESH_INSTALL,
    _VSCODE_EXT_INSTALLED,
    _VSCODE_EXT_VERSION,
)

from arrayview._vscode_signal import (  # noqa: F401
    _find_arrayview_window_id,
    _find_current_vscode_window_id,
    _open_direct_via_signal_file,
    _open_via_signal_file,
    _schedule_remote_open_retries,
    _write_vscode_signal,
    _VSCODE_COMPAT_SIGNAL_FILENAMES,
    _VSCODE_PORT_SETTINGS_SETTLE_SECONDS,
    _VSCODE_SIGNAL_FILENAME,
    _VSCODE_SIGNAL_MAX_AGE_MS,
)

from arrayview._vscode_shm import _ACTIVE_SHM, _open_direct_via_shm  # noqa: F401

from arrayview._vscode_browser import _open_browser, _print_viewer_location  # noqa: F401