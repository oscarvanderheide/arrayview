"""Best-effort, opt-in launch event tracing.

The trace is deliberately dependency-free and inert unless
``ARRAYVIEW_LAUNCH_TRACE`` names an absolute JSONL file.  Integrated-browser
route tracing can also be enabled explicitly with
``~/.arrayview/enable-launch-trace``.  Launch behavior must never depend on a
trace write succeeding.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


TRACE_PATH_ENV = "ARRAYVIEW_LAUNCH_TRACE"
TRACE_ID_ENV = "ARRAYVIEW_LAUNCH_ID"
TRACE_ROLE_ENV = "ARRAYVIEW_LAUNCH_ROLE"
TRACE_SCHEMA = 1
TRACE_MARKER_NAME = "enable-launch-trace"
TRACE_DEFAULT_NAME = "launch-trace.jsonl"
_MAX_EVENT_BYTES = 4096

_lock = threading.Lock()
_launch_id: str | None = None
_role: str | None = None
_path: str | None = None
_sequence = 0


def configure_launch_trace(
    *,
    path: str | None = None,
    launch_id: str | None = None,
    role: str | None = None,
) -> str | None:
    """Configure this process and return its launch ID when tracing is enabled."""
    candidate = path if path is not None else os.environ.get(TRACE_PATH_ENV)
    if not candidate or not Path(candidate).is_absolute():
        global _launch_id, _path, _role, _sequence
        with _lock:
            _path = None
            _launch_id = None
            _role = None
            _sequence = 0
        return None

    with _lock:
        _path = candidate
        _launch_id = launch_id or os.environ.get(TRACE_ID_ENV) or uuid.uuid4().hex
        _role = role or os.environ.get(TRACE_ROLE_ENV) or "parent"
        _sequence = 0
        return _launch_id


def trace_child_environment(base: dict[str, str] | None = None) -> dict[str, str] | None:
    """Return a copied environment carrying the active trace into a daemon."""
    if _path is None or _launch_id is None:
        return None
    child_env = dict(os.environ if base is None else base)
    child_env[TRACE_PATH_ENV] = _path
    child_env[TRACE_ID_ENV] = _launch_id
    child_env[TRACE_ROLE_ENV] = "daemon"
    return child_env


def trace_tag(value: object) -> str:
    """Return a stable, non-reversible tag suitable for trace correlation."""
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:12]


def configure_route_launch_trace() -> bool:
    """Enable route tracing from an explicit environment or marker opt-in.

    Route handlers call this directly so an already-running server can notice
    the marker even when its process environment never contained the trace
    variable.  An explicit environment value always wins, including an invalid
    relative value, which keeps the normal trace contract unambiguous.
    """
    try:
        explicit = os.environ.get(TRACE_PATH_ENV)
        if explicit is not None:
            if not explicit or not Path(explicit).is_absolute():
                return False
            if _path != explicit:
                configure_launch_trace()
            return _path == explicit

        trace_dir = Path.home() / ".arrayview"
        if not (trace_dir / TRACE_MARKER_NAME).is_file():
            return False
        candidate = str(trace_dir / TRACE_DEFAULT_NAME)
        if _path != candidate:
            configure_launch_trace(path=candidate, role="server")
        return _path == candidate
    except Exception:
        return False


def emit_route_launch_event(
    event: str,
    *,
    navigation_key: str,
    tab_key: str | None = None,
    **attrs: object,
) -> None:
    """Emit a route event without exposing private route credentials."""
    try:
        if not configure_route_launch_trace():
            return
        safe_attrs = {
            "navigation_key_tag": trace_tag(navigation_key),
            **attrs,
        }
        if tab_key is not None:
            safe_attrs["tab_key_tag"] = trace_tag(tab_key)
        emit_launch_event(event, **safe_attrs)
    except Exception:
        return


def emit_launch_event(event: str, **attrs: object) -> None:
    """Append one compact event, swallowing every tracing-related failure."""
    try:
        _ensure_environment_configuration()
        if _path is None or _launch_id is None or _role is None:
            return
        with _lock:
            global _sequence
            _sequence += 1
            payload = {
                "schema": TRACE_SCHEMA,
                "launch_id": _launch_id,
                "role": _role,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "seq": _sequence,
                "wall_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "event": event,
                "attrs": attrs,
            }
            encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            if len(encoded) > _MAX_EVENT_BYTES:
                payload["attrs"] = {"trace_error": "event_too_large"}
                encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            fd = os.open(_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, encoded)
            finally:
                os.close(fd)
    except Exception:
        return


def _ensure_environment_configuration() -> None:
    if _path is not None:
        return
    configure_launch_trace()


__all__ = [
    "TRACE_ID_ENV",
    "TRACE_MARKER_NAME",
    "TRACE_PATH_ENV",
    "TRACE_ROLE_ENV",
    "configure_launch_trace",
    "configure_route_launch_trace",
    "emit_launch_event",
    "emit_route_launch_event",
    "trace_child_environment",
    "trace_tag",
]
