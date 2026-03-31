"""Read/write ~/.arrayview/config.toml for persistent user preferences."""

from __future__ import annotations

import os
import sys

CONFIG_PATH = os.path.expanduser("~/.arrayview/config.toml")

_VALID_WINDOW_MODES = {"browser", "vscode", "native", "inline"}
_VALID_ENV_KEYS = {"default", "terminal", "vscode", "jupyter", "ssh", "julia"}


def load_config() -> dict:
    """Load config from TOML file. Returns {} on missing/malformed file."""
    if not os.path.isfile(CONFIG_PATH):
        return {}
    try:
        text = open(CONFIG_PATH).read()
        return _parse_toml(text)
    except Exception:
        return {}


def save_config(config: dict) -> None:
    """Write config dict to TOML file."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        f.write(_dump_toml(config))


def get_viewer_colormaps() -> list[str] | None:
    """Return user-configured colormap cycle list, or None if not configured."""
    cfg = load_config()
    viewer_cfg = cfg.get("viewer", {})
    if not isinstance(viewer_cfg, dict):
        return None
    colormaps = viewer_cfg.get("colormaps")
    if isinstance(colormaps, list) and all(isinstance(c, str) for c in colormaps) and colormaps:
        return colormaps
    return None


def get_nninteractive_url() -> str | None:
    """Return configured nnInteractive server URL, or None.

    Priority: ARRAYVIEW_NNINTERACTIVE_URL env var > config[nninteractive][url].
    """
    env_val = os.environ.get("ARRAYVIEW_NNINTERACTIVE_URL", "").strip()
    if env_val:
        return env_val
    cfg = load_config()
    nn_cfg = cfg.get("nninteractive", {})
    if isinstance(nn_cfg, dict):
        url = nn_cfg.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def get_window_default(environment: str) -> str | None:
    """Return the user's preferred window mode for the given environment.

    Priority: ARRAYVIEW_WINDOW env var > config[window][environment] > config[window][default].
    Returns None if no preference is set or the value is invalid.
    """
    env_val = os.environ.get("ARRAYVIEW_WINDOW", "").strip().lower()
    if env_val:
        return env_val if env_val in _VALID_WINDOW_MODES else None

    cfg = load_config()
    window_cfg = cfg.get("window", {})
    if not isinstance(window_cfg, dict):
        return None

    val = window_cfg.get(environment) or window_cfg.get("default")
    if isinstance(val, str) and val.strip().lower() in _VALID_WINDOW_MODES:
        return val.strip().lower()
    return None


# ---------------------------------------------------------------------------
# Minimal TOML parser/writer (no dependencies, handles our simple schema)
# ---------------------------------------------------------------------------


def _parse_toml(text: str) -> dict:
    """Parse simple TOML (flat tables with string values). Uses tomllib on 3.11+."""
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)
    # Minimal fallback for 3.10
    result: dict = {}
    current_table: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and not line.startswith("[["):
            key = line.strip("[] ")
            result[key] = {}
            current_table = result[key]
        elif "=" in line and current_table is not None:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            current_table[k] = v
    return result


def _dump_toml(config: dict) -> str:
    """Serialize config dict to TOML string."""
    lines: list[str] = []
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                lines.append(f'{k} = "{v}"')
            lines.append("")
    return "\n".join(lines) + "\n" if lines else ""
