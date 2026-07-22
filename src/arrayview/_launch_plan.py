"""Data-only launch routing primitives and environment snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
import socket
import sys
import time
import urllib.request
import uuid

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
    MATLAB = "matlab"
    VSCODE_EXPLORER = "vscode_explorer"
    CODEX = "codex"


class Environment(_StrEnum):
    TERMINAL = "terminal"
    VSCODE_LOCAL = "vscode_local"
    VSCODE_REMOTE = "vscode_remote"
    SSH = "ssh"
    JUPYTER = "jupyter"
    JULIA = "julia"
    MATLAB = "matlab"


class Placement(_StrEnum):
    """Where the caller can actually observe a display."""

    LOCAL = "local"
    VSCODE_LOCAL = "vscode_local"
    VSCODE_REMOTE = "vscode_remote"
    SSH = "ssh"


class CallerScope(_StrEnum):
    CLI = "cli"
    SCRIPT = "script"
    INTERACTIVE = "interactive"
    KERNEL = "kernel"
    EMBEDDED = "embedded"
    EXPLORER = "explorer"


class Transport(_StrEnum):
    HTTP = "http"
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
    RELAY = "relay"


class LaunchFailure(_StrEnum):
    """Stable failure codes returned by :func:`plan_launch`."""

    INVALID_PORT = "invalid_port"
    INVALID_WINDOW = "invalid_window"
    VSCODE_UNAVAILABLE = "vscode_unavailable"
    REMOTE_PORT_CONFLICT = "remote_port_conflict"


class CompletionTarget(_StrEnum):
    """Observable milestone at which the invocation may return successfully."""

    SESSION_ACCEPTED = "session_accepted"
    DISPLAY_ACCEPTED = "display_accepted"
    DISPATCH_ACCEPTED = "dispatch_accepted"
    FRAME_READY = "frame_ready"
    MIME_RETURNED_OR_EMITTED = "mime_returned_or_emitted"
    DISPLAY_SIDE_EFFECT_EMITTED = "display_side_effect_emitted"
    GUIDANCE_PRINTED = "guidance_printed"


@dataclass(frozen=True)
class ServerSnapshot:
    port: int
    port_busy: bool
    arrayview_server_alive: bool
    server_pid: int | None = None
    server_hostname: str | None = None
    server_instance_id: str | None = None
    server_process_start: str | None = None
    server_capabilities: tuple[str, ...] = ()
    server_protocol_version: str | None = None


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
    cli_default_server: ServerSnapshot | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LaunchRequest:
    port: int
    requested_window: str | None = None


@dataclass(frozen=True)
class LaunchIntent:
    """Normalized user intent shared by CLI and Python entry points."""

    invocation: Invocation
    port: int
    requested_window: str | None = None
    browser: bool = False
    inline: bool | None = None
    window_explicit: bool = False
    inline_explicit: bool = False
    persistent: bool = False


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
    requested_port: int = 8123
    effective_port: int = 8123
    reasons: tuple[str, ...] = ()
    failure: LaunchFailure | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation suitable for diagnostics."""
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class LaunchContext:
    """Immutable selection authority carried through one launch execution.

    Dynamic resources may be revalidated after this is created, but downstream
    code must not re-detect the host or choose a different display adapter.
    """

    launch_id: str
    intent: LaunchIntent
    evidence: LaunchEnvironmentSnapshot
    plan: LaunchPlan
    placement: Placement
    caller_scope: CallerScope
    completion_target: CompletionTarget

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


def create_launch_context(
    intent: LaunchIntent,
    evidence: LaunchEnvironmentSnapshot | None = None,
    *,
    launch_id: str | None = None,
    caller_scope: CallerScope | str | None = None,
) -> LaunchContext:
    """Capture facts once and bind them to the resulting immutable plan."""
    snapshot = evidence or snapshot_launch_environment(
        intent.port, intent.invocation, requested_window=intent.requested_window
    )
    plan = plan_launch(intent, snapshot)
    return LaunchContext(
        launch_id=(
            launch_id
            or os.environ.get("ARRAYVIEW_LAUNCH_ID")
            or uuid.uuid4().hex
        ),
        intent=intent,
        evidence=snapshot,
        plan=plan,
        placement=_placement(snapshot),
        caller_scope=(
            CallerScope(caller_scope)
            if caller_scope is not None
            else _caller_scope(intent.invocation, snapshot)
        ),
        completion_target=_completion_target(intent.invocation, plan, snapshot),
    )


def _completion_target(
    invocation: Invocation,
    plan: LaunchPlan,
    evidence: LaunchEnvironmentSnapshot,
) -> CompletionTarget:
    if plan.display is Display.NONE:
        return CompletionTarget.SESSION_ACCEPTED
    if plan.display is Display.INLINE:
        if invocation is Invocation.JULIA:
            return CompletionTarget.DISPLAY_SIDE_EFFECT_EMITTED
        return CompletionTarget.MIME_RETURNED_OR_EMITTED
    if plan.display is Display.BROWSER:
        if _placement(evidence) is Placement.SSH:
            return CompletionTarget.GUIDANCE_PRINTED
        return CompletionTarget.DISPATCH_ACCEPTED
    if invocation in {Invocation.CLI, Invocation.VSCODE_EXPLORER}:
        return CompletionTarget.FRAME_READY
    return CompletionTarget.DISPLAY_ACCEPTED


def _placement(evidence: LaunchEnvironmentSnapshot) -> Placement:
    if evidence.is_vscode_remote:
        return Placement.VSCODE_REMOTE
    if evidence.in_vscode_terminal:
        return Placement.VSCODE_LOCAL
    if evidence.ssh_connection or evidence.ssh_client:
        return Placement.SSH
    return Placement.LOCAL


def _caller_scope(
    invocation: Invocation, evidence: LaunchEnvironmentSnapshot
) -> CallerScope:
    if invocation is Invocation.CLI:
        return CallerScope.CLI
    if invocation is Invocation.VSCODE_EXPLORER:
        return CallerScope.EXPLORER
    if invocation is Invocation.JUPYTER or evidence.in_jupyter:
        return CallerScope.KERNEL
    if invocation in {Invocation.JULIA, Invocation.MATLAB}:
        return CallerScope.EMBEDDED
    return CallerScope.INTERACTIVE


def plan_launch(
    intent: LaunchIntent, facts: LaunchEnvironmentSnapshot
) -> LaunchPlan:
    """Compute launch policy without starting servers or opening displays.

    ``facts`` is deliberately supplied by the caller: planning is deterministic
    and can be exercised without probing the machine or importing server code.
    """
    reasons: list[str] = []
    port = intent.port
    if not 1 <= port <= 65535:
        return _failed_plan(
            intent, facts, LaunchFailure.INVALID_PORT, "port_out_of_range"
        )

    raw_window = intent.requested_window
    window = raw_window.strip().lower() if isinstance(raw_window, str) else None
    if intent.window_explicit and raw_window is None:
        window = "browser"
        reasons.append("explicit_window_false")
    valid_windows = {None, "native", "browser", "vscode", "inline", "none"}
    if window not in valid_windows:
        return _failed_plan(
            intent, facts, LaunchFailure.INVALID_WINDOW, "unknown_window"
        )

    if intent.browser and window is None:
        window = "browser"
        reasons.append("browser_flag")
    if (
        window is None
        and facts.config_default
        and not intent.browser
        and not intent.window_explicit
        and not intent.inline_explicit
    ):
        window = facts.config_default.strip().lower()
        reasons.append("config_window_default")
    if window not in valid_windows:
        return _failed_plan(
            intent, facts, LaunchFailure.INVALID_WINDOW, "invalid_config_window"
        )

    environment = facts.environment
    explicit_inline = window == "inline" or (
        intent.inline is True and intent.inline_explicit
    )
    remote_jupyter = (
        facts.is_vscode_remote
        and facts.in_jupyter
        and intent.inline is not False
        and not explicit_inline
    )
    if remote_jupyter:
        environment = Environment.VSCODE_REMOTE
        reasons.append("remote_jupyter_uses_vscode")
    elif intent.invocation is Invocation.JUPYTER or facts.in_jupyter:
        environment = Environment.JUPYTER
    elif intent.invocation is Invocation.JULIA or facts.in_julia:
        environment = Environment.JULIA
    elif intent.invocation is Invocation.MATLAB and not facts.is_vscode_remote:
        environment = Environment.MATLAB

    server = facts.server
    if (
        intent.invocation is Invocation.PYTHON
        and intent.port == 8123
        and environment is Environment.VSCODE_REMOTE
        and facts.cli_default_server is not None
        and _server_reusable(facts.cli_default_server)
    ):
        port = facts.cli_default_server.port
        server = facts.cli_default_server
        reasons.append("reuse_remote_cli_default_server")

    if server.arrayview_server_alive and not _server_reusable(server):
        port += 1
        reasons.append("incompatible_server_new_port")
        server = ServerSnapshot(port, False, False)

    busy_foreign = server.port_busy and not server.arrayview_server_alive
    if busy_foreign and environment is Environment.VSCODE_REMOTE:
        return _failed_plan(
            intent, facts, LaunchFailure.REMOTE_PORT_CONFLICT,
            "remote_forwarding_requires_requested_port", effective_port=port,
        )
    if busy_foreign:
        port += 1
        reasons.append("scan_from_next_port")

    if _server_reusable(server):
        owner = ServerOwner.EXISTING
        registration = Registration.HTTP_LOAD
        reasons.append("reuse_compatible_server")
    elif environment is Environment.JULIA:
        owner = ServerOwner.SPAWNED_DAEMON
        registration = Registration.DAEMON_STARTUP
        reasons.append(f"{environment.value}_requires_subprocess")
    elif intent.invocation in {Invocation.CLI, Invocation.VSCODE_EXPLORER}:
        owner = (
            ServerOwner.PERSISTENT
            if intent.persistent
            else ServerOwner.SPAWNED_DAEMON
        )
        registration = Registration.DAEMON_STARTUP
        reasons.append("cli_server_process")
    else:
        owner = ServerOwner.PERSISTENT if intent.persistent else ServerOwner.IN_PROCESS
        registration = Registration.IN_PROCESS_SESSION
        reasons.append("python_process_owns_session")

    display, fallback, fallback_allowed = _display_policy(
        window=None if remote_jupyter else window,
        inline=False if remote_jupyter else intent.inline,
        environment=environment,
        native_available=facts.native_backend is not None, reasons=reasons,
    )
    if display is Display.VSCODE and environment not in {
        Environment.VSCODE_LOCAL, Environment.VSCODE_REMOTE
    }:
        return _failed_plan(
            intent, facts, LaunchFailure.VSCODE_UNAVAILABLE,
            "vscode_display_requested_outside_vscode", effective_port=port,
        )
    return LaunchPlan(
        invocation=intent.invocation, environment=environment,
        transport=Transport.HTTP, server_owner=owner, display=display,
        registration=registration, fallback_display=fallback,
        fallback_allowed=fallback_allowed, requested_port=intent.port,
        effective_port=port, reasons=tuple(reasons),
    )


def _display_policy(
    *,
    window: str | None,
    inline: bool | None,
    environment: Environment,
    native_available: bool,
    reasons: list[str],
) -> tuple[Display, Display | None, bool]:
    if window == "none":
        reasons.append("display_disabled")
        return Display.NONE, None, False
    if (
        window == "inline"
        or inline is True
        or (
            window is None
            and inline is None
            and environment is Environment.JUPYTER
        )
    ):
        reasons.append("jupyter_inline" if window is None else "explicit_inline")
        return Display.INLINE, None, False
    if window == "vscode" or (
        window is None
        and environment in {Environment.VSCODE_LOCAL, Environment.VSCODE_REMOTE}
    ):
        reasons.append("vscode_environment" if window is None else "explicit_vscode")
        if environment is Environment.VSCODE_REMOTE:
            return Display.VSCODE, None, False
        return Display.VSCODE, Display.BROWSER, True
    if window == "browser":
        if environment is Environment.VSCODE_REMOTE:
            reasons.append("remote_browser_redirected_to_vscode")
            return Display.VSCODE, None, False
        reasons.append("explicit_browser")
        return Display.BROWSER, None, False
    if window == "native":
        if environment is Environment.VSCODE_REMOTE:
            reasons.append("remote_native_redirected_to_vscode")
            return Display.VSCODE, None, False
        if native_available:
            reasons.append("explicit_native")
            return Display.NATIVE, Display.BROWSER, True
        reasons.append("native_unavailable")
        return Display.BROWSER, None, False
    if native_available and environment in {Environment.TERMINAL, Environment.MATLAB}:
        reasons.append("native_available")
        return Display.NATIVE, Display.BROWSER, True
    reasons.append("browser_default")
    return Display.BROWSER, None, False


def _failed_plan(
    intent: LaunchIntent,
    facts: LaunchEnvironmentSnapshot,
    failure: LaunchFailure,
    reason: str,
    *,
    effective_port: int | None = None,
) -> LaunchPlan:
    return LaunchPlan(
        invocation=intent.invocation, environment=facts.environment,
        transport=Transport.NONE, server_owner=ServerOwner.EXTERNAL,
        display=Display.NONE, registration=Registration.RELAY,
        requested_port=intent.port,
        effective_port=intent.port if effective_port is None else effective_port,
        reasons=(reason,), failure=failure,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


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
        cli_default_server=(
            _server_snapshot(8000)
            if inv is Invocation.PYTHON
            and environment is Environment.VSCODE_REMOTE
            and port == 8123
            else None
        ),
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
        server_instance_id=(
            _str_or_none(payload.get("instance_id")) if payload else None
        ),
        server_process_start=(
            _str_or_none(payload.get("process_start")) if payload else None
        ),
        server_capabilities=(
            tuple(
                item
                for item in payload.get("capabilities", [])
                if isinstance(item, str)
            )
            if payload
            else ()
        ),
        server_protocol_version=(
            _str_or_none(payload.get("protocol_version")) if payload else None
        ),
    )


def _server_reusable(server: ServerSnapshot) -> bool:
    return (
        server.arrayview_server_alive
        and server.server_instance_id is not None
        and server.server_protocol_version == "1"
        and "identity-fenced-load" in server.server_capabilities
        and "identity-fenced-mutations" in server.server_capabilities
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
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=_PING_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    raise RuntimeError("unexpected health status")
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return payload
        except Exception:
            pass
        if attempt < 2:
            time.sleep(0.05)
    return None


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
