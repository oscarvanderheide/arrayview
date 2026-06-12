"""Data-only launch routing primitives and environment snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
import socket
import sys
import urllib.request

_LOOPBACK_HOST = "localhost"
_PING_TIMEOUT_SECONDS = 0.2

_SNAPSHOT_ENV_KEYS = (
    "ARRAYVIEW_WINDOW",
    "ARRAYVIEW_WINDOW_ID",
    "DISPLAY",
    "TERM_PROGRAM",
    "VSCODE_AGENT_FOLDER",
    "VSCODE_INJECTION",
    "VSCODE_IPC_HOOK_CLI",
    "WAYLAND_DISPLAY",
    "SSH_CLIENT",
    "SSH_CONNECTION",
)


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Invocation(_StrEnum):
    CLI = "cli"
    PYTHON = "python"
    JUPYTER = "jupyter"
    JULIA = "julia"
    STDIO = "stdio"
    CODEX = "codex"


class Environment(_StrEnum):
    TERMINAL = "terminal"
    VSCODE_LOCAL = "vscode_local"
    VSCODE_REMOTE = "vscode_remote"
    SSH = "ssh"
    JUPYTER = "jupyter"
    JULIA = "julia"


class Transport(_StrEnum):
    HTTP = "http"
    STDIO_FILE = "stdio_file"
    STDIO_SHM = "stdio_shm"
    NONE = "none"


class ServerOwner(_StrEnum):
    EXISTING = "existing"
    SPAWNED_DAEMON = "spawned_daemon"
    IN_PROCESS = "in_process"
    PERSISTENT = "persistent"
    EXTERNAL = "external"


class Display(_StrEnum):
    NATIVE = "native"
    BROWSER = "browser"
    VSCODE = "vscode"
    INLINE = "inline"
    NONE = "none"


class Registration(_StrEnum):
    HTTP_LOAD = "http_load"
    DAEMON_STARTUP = "daemon_startup"
    IN_PROCESS_SESSION = "in_process_session"
    STDIO_REGISTER = "stdio_register"
    RELAY = "relay"


@dataclass(frozen=True)
class ServerSnapshot:
    port: int
    port_busy: bool
    arrayview_server_alive: bool
    server_pid: int | None = None
    server_hostname: str | None = None


@dataclass(frozen=True)
class LaunchEnvironmentSnapshot:
    invocation: Invocation
    requested_window: str | None
    environment: Environment
    platform: str
    env_vars: dict[str, str]
    config_default: str | None
    native_backend: str | None
    server: ServerSnapshot
    in_jupyter: bool
    in_julia: bool
    in_vscode_terminal: bool
    is_vscode_remote: bool
    in_vscode_tunnel: bool
    ssh_connection: bool
    ssh_client: bool
    hostname: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LaunchRequest:
    port: int
    requested_window: str | None = None


@dataclass(frozen=True)
class LaunchPlan:
    invocation: Invocation
    environment: Environment
    transport: Transport
    server_owner: ServerOwner
    display: Display
    registration: Registration
    fallback_display: Display | None = None
    fallback_allowed: bool = False


def snapshot_launch_environment(
    port: int,
    invocation: Invocation | str,
    requested_window: str | None = None,
) -> LaunchEnvironmentSnapshot:
    """Capture launch-relevant facts without importing the server stack."""
    inv = _coerce_invocation(invocation)
    env_vars = _snapshot_env_vars()
    in_jupyter = _platform_bool("_in_jupyter")
    in_julia = _platform_bool("_is_julia_env")
    in_vscode_terminal = _platform_bool("_in_vscode_terminal")
    is_vscode_remote = _platform_bool("_is_vscode_remote")
    in_vscode_tunnel = _platform_bool("_in_vscode_tunnel")
    ssh_connection = bool(os.environ.get("SSH_CONNECTION"))
    ssh_client = bool(os.environ.get("SSH_CLIENT"))
    environment = _classify_environment(
        in_jupyter=in_jupyter,
        in_julia=in_julia,
        in_vscode_terminal=in_vscode_terminal,
        is_vscode_remote=is_vscode_remote,
        ssh_connection=ssh_connection,
        ssh_client=ssh_client,
    )

    return LaunchEnvironmentSnapshot(
        invocation=inv,
        requested_window=requested_window,
        environment=environment,
        platform=sys.platform,
        env_vars=env_vars,
        config_default=_config_window_default(environment.value),
        native_backend=_native_window_gui(),
        server=_server_snapshot(port),
        in_jupyter=in_jupyter,
        in_julia=in_julia,
        in_vscode_terminal=in_vscode_terminal,
        is_vscode_remote=is_vscode_remote,
        in_vscode_tunnel=in_vscode_tunnel,
        ssh_connection=ssh_connection,
        ssh_client=ssh_client,
        hostname=socket.gethostname(),
    )


def _coerce_invocation(value: Invocation | str) -> Invocation:
    if isinstance(value, Invocation):
        return value
    return Invocation(value)


def _snapshot_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in _SNAPSHOT_ENV_KEYS if key in os.environ}


def _platform_bool(name: str) -> bool:
    try:
        from arrayview import _platform

        return bool(getattr(_platform, name)())
    except Exception:
        return False


def _native_window_gui() -> str | None:
    try:
        from arrayview._platform import _native_window_gui as native_window_gui

        return native_window_gui()
    except Exception:
        return None


def _config_window_default(environment: str) -> str | None:
    try:
        from arrayview._config import load_config

        cfg = load_config()
    except Exception:
        return None
    window_cfg = cfg.get("window", {})
    if not isinstance(window_cfg, dict):
        return None
    env_key = "vscode" if environment in {"vscode_local", "vscode_remote"} else environment
    value = window_cfg.get(env_key) or window_cfg.get("default")
    if isinstance(value, str):
        return value.strip().lower() or None
    return None


def _classify_environment(
    *,
    in_jupyter: bool,
    in_julia: bool,
    in_vscode_terminal: bool,
    is_vscode_remote: bool,
    ssh_connection: bool,
    ssh_client: bool,
) -> Environment:
    if in_jupyter:
        return Environment.JUPYTER
    if in_julia:
        return Environment.JULIA
    if is_vscode_remote:
        return Environment.VSCODE_REMOTE
    if in_vscode_terminal:
        return Environment.VSCODE_LOCAL
    if ssh_connection or ssh_client:
        return Environment.SSH
    return Environment.TERMINAL


def _server_snapshot(port: int) -> ServerSnapshot:
    payload = _ping_arrayview_server(port)
    return ServerSnapshot(
        port=port,
        port_busy=_port_busy(port),
        arrayview_server_alive=payload is not None,
        server_pid=_int_or_none(payload.get("pid")) if payload else None,
        server_hostname=_str_or_none(payload.get("hostname")) if payload else None,
    )


def _port_busy(port: int) -> bool:
    try:
        with socket.create_connection(
            (_LOOPBACK_HOST, port), timeout=_PING_TIMEOUT_SECONDS
        ):
            return True
    except OSError:
        return False


def _ping_arrayview_server(port: int) -> dict | None:
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=_PING_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("ok") is True and payload.get("service") == "arrayview":
        return payload
    return None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
