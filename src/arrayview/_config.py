"""Read/write ~/.arrayview/config.toml for persistent user preferences."""

from __future__ import annotations

import datetime as _datetime
import math
import os
import tempfile
import threading
from copy import deepcopy

CONFIG_PATH = os.path.expanduser("~/.arrayview/config.toml")

_VALID_WINDOW_MODES = {"browser", "vscode", "native", "inline", "none"}
_VALID_ENV_KEYS = {"default", "terminal", "vscode", "jupyter", "ssh", "julia"}
_VALID_THEMES = {"dark", "light"}
_VALID_ORTHO_LAYOUTS = {"horizontal", "big-left"}
_VALID_DIMBAR_MODES = {"compact", "extended"}
_CONFIG_LOCK = threading.Lock()

WINDOW_PREFERENCE_OPTIONS = {
    "default": ["native", "browser", "none"],
    "terminal": ["native", "browser", "none"],
    "vscode": ["vscode", "browser", "native", "none"],
    "jupyter": ["inline", "browser", "native", "none"],
    "ssh": ["browser", "none"],
    "julia": ["browser", "native", "none"],
}
BUILTIN_PREFERENCES = {
    "window": {
        "default": "native",
        "terminal": "native",
        "vscode": "vscode",
        "jupyter": "inline",
        "ssh": "browser",
        "julia": "browser",
    },
    "viewer": {
        "theme": "dark",
        "rounded_panes": True,
        "ortho_layout": None,
        "dimbar_mode": "compact",
    },
}
PREFERENCE_SCHEMA = {
    "window": WINDOW_PREFERENCE_OPTIONS,
    "viewer": {
        "theme": sorted(_VALID_THEMES),
        "rounded_panes": "boolean",
        "colormaps": "non-empty list of names",
        "ortho_layout": sorted(_VALID_ORTHO_LAYOUTS),
        "dimbar_mode": sorted(_VALID_DIMBAR_MODES),
    },
    "nninteractive": {"url": "string"},
}


class MalformedConfigError(ValueError):
    """Raised when an update would overwrite an unreadable config file."""


class InvalidPreferenceError(ValueError):
    """Raised when an update contains an unknown or invalid preference."""


def _load_config_strict() -> dict:
    if not os.path.isfile(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            parsed = _parse_toml(f.read())
    except Exception as exc:
        raise MalformedConfigError("The existing config.toml is malformed") from exc
    if not isinstance(parsed, dict):
        raise MalformedConfigError("The existing config.toml is not a TOML table")
    return parsed


def load_config() -> dict:
    """Load config from TOML file. Returns {} on missing/malformed file."""
    try:
        return _load_config_strict()
    except MalformedConfigError:
        return {}


def save_config(config: dict) -> None:
    """Atomically write a config dict to its TOML file."""
    folder = os.path.dirname(CONFIG_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    payload = _dump_toml(config)
    fd, temp_path = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=folder or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, CONFIG_PATH)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _normalized_string(value, allowed: set[str], label: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise InvalidPreferenceError(f"{label} must be one of: {', '.join(sorted(allowed))}")
    return value.strip().lower()


def _validate_preference(section: str, key: str, value):
    if section == "window" and key in _VALID_ENV_KEYS:
        return _normalized_string(
            value, set(WINDOW_PREFERENCE_OPTIONS[key]), f"window.{key}"
        )
    if section == "viewer" and key == "theme":
        return _normalized_string(value, _VALID_THEMES, "viewer.theme")
    if section == "viewer" and key == "rounded_panes":
        if not isinstance(value, bool):
            raise InvalidPreferenceError("viewer.rounded_panes must be a boolean")
        return value
    if section == "viewer" and key == "colormaps":
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise InvalidPreferenceError("viewer.colormaps must be a non-empty list of names")
        normalized = [item.strip() for item in value]
        if len(set(normalized)) != len(normalized):
            raise InvalidPreferenceError("viewer.colormaps must not contain duplicates")
        return normalized
    if section == "viewer" and key == "ortho_layout":
        return _normalized_string(value, _VALID_ORTHO_LAYOUTS, "viewer.ortho_layout")
    if section == "viewer" and key == "dimbar_mode":
        return _normalized_string(value, _VALID_DIMBAR_MODES, "viewer.dimbar_mode")
    if section == "nninteractive" and key == "url":
        if not isinstance(value, str) or not value.strip():
            raise InvalidPreferenceError("nninteractive.url must be a non-empty string")
        return value.strip()
    raise InvalidPreferenceError(f"Unknown preference: {section}.{key}")


def update_preferences(changes: dict) -> dict:
    """Validate and merge preferences into the latest config on disk.

    ``None`` removes a known preference, restoring its built-in default.
    Unknown config sections and keys already on disk are preserved.
    """
    if not isinstance(changes, dict) or not changes:
        raise InvalidPreferenceError("Preference changes must be a non-empty object")
    validated: dict[str, dict] = {}
    for section, values in changes.items():
        if section not in PREFERENCE_SCHEMA or not isinstance(values, dict):
            raise InvalidPreferenceError(f"Unknown preference section: {section}")
        if not values:
            raise InvalidPreferenceError(f"Preference section is empty: {section}")
        validated[section] = {}
        for key, value in values.items():
            if key not in PREFERENCE_SCHEMA[section]:
                raise InvalidPreferenceError(f"Unknown preference: {section}.{key}")
            validated[section][key] = None if value is None else _validate_preference(section, key, value)

    with _CONFIG_LOCK:
        config = _load_config_strict()
        merged = deepcopy(config)
        for section, values in validated.items():
            target = merged.get(section)
            if target is None:
                target = {}
                merged[section] = target
            if not isinstance(target, dict):
                raise MalformedConfigError(f"Config section [{section}] is not a table")
            for key, value in values.items():
                if value is None:
                    target.pop(key, None)
                else:
                    target[key] = value
            if not target:
                merged.pop(section, None)
        save_config(merged)
        return merged


def get_preferences() -> dict:
    """Return only supported preferences currently present in the config."""
    config = load_config()
    result: dict[str, dict] = {}
    for section, keys in PREFERENCE_SCHEMA.items():
        source = config.get(section)
        if not isinstance(source, dict):
            continue
        selected = {key: source[key] for key in keys if key in source}
        if selected:
            result[section] = selected
    return result


def get_viewer_colormaps() -> list[str] | None:
    """Return user-configured colormap cycle list, or None if not configured."""
    viewer_cfg = load_config().get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    colormaps = viewer_cfg.get("colormaps")
    if isinstance(colormaps, list) and all(isinstance(c, str) for c in colormaps) and colormaps:
        return colormaps
    return None


def get_viewer_theme() -> str | None:
    """Return user-configured default theme name, or None if not configured."""
    viewer_cfg = load_config().get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    theme = viewer_cfg.get("theme")
    if isinstance(theme, str) and theme.strip().lower() in _VALID_THEMES:
        return theme.strip().lower()
    return None


def get_viewer_rounded_panes() -> bool | None:
    """Return user-configured default for rounded panes, or None if not configured."""
    viewer_cfg = load_config().get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    val = viewer_cfg.get("rounded_panes")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return None


def get_viewer_ortho_layout() -> str | None:
    """Return the preferred initial ortho-view layout."""
    viewer_cfg = load_config().get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    value = viewer_cfg.get("ortho_layout")
    return value.strip().lower() if isinstance(value, str) and value.strip().lower() in _VALID_ORTHO_LAYOUTS else None


def get_viewer_dimbar_mode() -> str | None:
    """Return the preferred initial dimbar mode."""
    viewer_cfg = load_config().get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    value = viewer_cfg.get("dimbar_mode")
    return value.strip().lower() if isinstance(value, str) and value.strip().lower() in _VALID_DIMBAR_MODES else None


def get_nninteractive_url() -> str | None:
    """Return configured nnInteractive server URL, or None."""
    env_val = os.environ.get("ARRAYVIEW_NNINTERACTIVE_URL", "").strip()
    if env_val:
        return env_val
    nn_cfg = load_config().get("nninteractive", {})
    if isinstance(nn_cfg, dict):
        url = nn_cfg.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def get_window_default(environment: str) -> str | None:
    """Return the user's preferred window mode for the given environment."""
    env_val = os.environ.get("ARRAYVIEW_WINDOW", "").strip().lower()
    if env_val:
        return env_val if env_val in _VALID_WINDOW_MODES else None
    window_cfg = load_config().get("window", {})
    if not isinstance(window_cfg, dict):
        return None
    val = window_cfg.get(environment) or window_cfg.get("default")
    if isinstance(val, str) and val.strip().lower() in _VALID_WINDOW_MODES:
        return val.strip().lower()
    return None


def _parse_toml(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def _toml_key(value: str) -> str:
    if value and all(
        char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in value
    ):
        return value
    return _toml_string(value)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\b", "\\b").replace("\t", "\\t").replace("\n", "\\n")
    escaped = escaped.replace("\f", "\\f").replace("\r", "\\r")
    return f'"{escaped}"'


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value).lower()
        return repr(value)
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _dump_toml(config: dict) -> str:
    """Serialize nested dictionaries and scalar/list TOML values."""
    lines: list[str] = []

    def emit_table(path: tuple[str, ...], table: dict) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if path:
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, value in scalar_items:
            lines.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
        if path and (scalar_items or not child_items):
            lines.append("")
        for key, value in child_items:
            emit_table((*path, str(key)), value)

    if not isinstance(config, dict):
        raise TypeError("Config must be a dictionary")
    root_scalars = {key: value for key, value in config.items() if not isinstance(value, dict)}
    root_tables = {key: value for key, value in config.items() if isinstance(value, dict)}
    for key, value in root_scalars.items():
        lines.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
    if root_scalars and root_tables:
        lines.append("")
    for key, value in root_tables.items():
        emit_table((str(key),), value)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")
