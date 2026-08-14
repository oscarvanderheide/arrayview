"""Extra listening ports on the already-running server.

Why this exists
---------------

In VS Code the viewer is fetched over a forwarded port.  Measured 2026-08-06 on
the real host: a forward that VS Code has *just created* never loses a page
request (4/4, and 3/3 on a genuine cold start), while one that already exists
and has sat idle for tens of seconds drops the first two to four.  A dropped
request never arrives — the opener can only discard the tab and open another,
which the user sees as the viewer flickering before the array appears.

While any viewer is open the server keeps the forward from going idle (see the
nudge in ``_routes_websocket``).  The case that leaves is the cold one: no
viewer is open, so nothing has kept the forward alive, and the first launch pays.

A port that has never been forwarded does not have that problem.  So a cold
launch is served from a fresh port, which VS Code forwards for the first time.
This module adds that port to the process that is *already* running, rather than
starting another server: same process, same loaded arrays, nothing extra to shut
down, and overlapping opens share the same private viewer port.

Lifetime
--------

Every launch leases the singleton viewer port before its URL is handed off.
The port is closed only when no viewer is connected **and** no launch still has
a live lease.  This prevents the last old viewer from closing the port while a
new viewer is between URL handoff and WebSocket connection.

It must not be closed while a viewer loaded from it is alive: the viewer keeps
using HTTP on its own origin long after the page has rendered, so pulling the
port out from under it would break the viewer rather than tidy up after it.

A launch that never receives a viewer loses its lease after a bounded timeout.
Without that, a failed handoff would leak a port with nothing to trigger its
release.
"""

from __future__ import annotations

import asyncio
import socket
import time

import arrayview._session as _session_mod
from arrayview._session import _vprint

# A lease must be long enough for a slow or temporarily wedged VS Code handoff,
# but a caller may not keep a private port alive without bound.
MIN_LEASE_TTL_MS = 1
MAX_LEASE_TTL_MS = 300_000
# Keep the origin alive across the viewer's ordinary reconnect window.  The
# correlated session uses the same 30-second default before it treats a lost
# WebSocket as a closed tab; closing the listener sooner would make that
# recovery impossible even though the server and session are still healthy.
VIEWER_RECONNECT_GRACE_S = 30.0

# port -> {"socket", "server", "task", "viewers", "opened_at", "leases",
#          "idle_until", "idle_generation"}
_EXTRA_PORTS: dict[int, dict] = {}
_PORT_LOCK = asyncio.Lock()


def extra_ports() -> list[int]:
    """Ports currently being served in addition to the main one."""
    return sorted(_EXTRA_PORTS)


def _bind_free_port() -> socket.socket:
    """Bind a port the OS says is free, so nothing has to be guessed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sock.listen(128)
    return sock


def _start_extra_port() -> tuple[int, dict] | None:
    """Start the singleton extra listener; caller holds ``_PORT_LOCK``."""
    try:
        import uvicorn

        from arrayview import _server as _server_mod

        sock = _bind_free_port()
        port = sock.getsockname()[1]
        config = uvicorn.Config(
            _server_mod.app,
            log_level="error",
            timeout_keep_alive=30,
            ws_ping_interval=None,
            ws="websockets",
            ws_per_message_deflate=True,
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(sockets=[sock]))
        entry = {
            "socket": sock,
            "server": server,
            "task": task,
            "viewers": 0,
            "opened_at": time.monotonic(),
            "leases": {},
            "idle_until": 0.0,
            "idle_generation": 0,
        }
        # The socket is already listening and accepting before serve() runs, so
        # a client that connects immediately is queued rather than refused.
        _vprint(f"[ArrayView] serving the private viewer port on {port}")
        return port, entry
    except Exception as exc:  # pragma: no cover - defensive
        _vprint(f"[ArrayView] could not add the private viewer port: {exc}")
        return None


async def acquire_viewer_port(request_id: str, ttl_ms: int) -> tuple[int | None, bool]:
    """Lease the singleton private viewer port for one launch.

    The selection and lease insertion are one event-loop transaction, so two
    launches arriving before either viewer connects cannot create two ports.
    """
    ttl_ms = max(MIN_LEASE_TTL_MS, min(MAX_LEASE_TTL_MS, int(ttl_ms)))
    retired: list[tuple[int, dict]] = []
    async with _PORT_LOCK:
        now = time.monotonic()
        for port, entry in list(_EXTRA_PORTS.items()):
            leases = entry.setdefault("leases", {})
            for lease_id, expires_at in list(leases.items()):
                if expires_at <= now:
                    leases.pop(lease_id, None)
            if (
                entry["viewers"] == 0
                and not leases
                and float(entry.get("idle_until", 0.0)) <= now
            ):
                _EXTRA_PORTS.pop(port, None)
                retired.append((port, entry))
        reused = bool(_EXTRA_PORTS)
        if reused:
            port = next(iter(_EXTRA_PORTS))
            entry = _EXTRA_PORTS[port]
        else:
            started = _start_extra_port()
            if started is None:
                for retired_port, retired_entry in retired:
                    asyncio.create_task(
                        _shutdown_entry(retired_port, retired_entry)
                    )
                return None, False
            port, entry = started
            _EXTRA_PORTS[port] = entry
        expires_at = now + ttl_ms / 1000.0
        entry.setdefault("leases", {})[request_id] = expires_at
        entry["idle_until"] = 0.0
        entry["idle_generation"] = int(entry.get("idle_generation", 0)) + 1
        asyncio.create_task(_expire_lease(port, request_id, expires_at))
    for retired_port, retired_entry in retired:
        asyncio.create_task(_shutdown_entry(retired_port, retired_entry))
    return port, reused


def viewer_connected(port: int | None, request_id: str | None = None) -> None:
    """Note that a viewer arrived and consume only its correlated lease."""
    entry = _EXTRA_PORTS.get(port)
    if entry is None:
        return
    leases = entry.setdefault("leases", {})
    if request_id:
        consumed = leases.pop(request_id, None)
        # Opener 0.15.47 asks for the port before it knows how to lease it.
        # The endpoint gives that request a bounded compatibility lease so a
        # package update does not force an immediate window reload. Consume one
        # such lease only when no exact lease exists.
        if consumed is None:
            compatibility_lease = next(
                (
                    lease_id
                    for lease_id in leases
                    if lease_id.startswith("compat-")
                ),
                None,
            )
            if compatibility_lease is not None:
                leases.pop(compatibility_lease, None)
    entry["viewers"] += 1
    entry["idle_until"] = 0.0
    entry["idle_generation"] = int(entry.get("idle_generation", 0)) + 1


async def viewer_disconnected(port: int | None) -> None:
    """Release a port only after its last viewer and last handoff lease."""
    close_after: tuple[int, float] | None = None
    async with _PORT_LOCK:
        entry = _EXTRA_PORTS.get(port)
        if entry is None:
            return
        entry["viewers"] = max(0, entry["viewers"] - 1)
        if entry["viewers"] == 0:
            entry["idle_generation"] = int(entry.get("idle_generation", 0)) + 1
            generation = entry["idle_generation"]
            idle_until = time.monotonic() + VIEWER_RECONNECT_GRACE_S
            entry["idle_until"] = idle_until
            close_after = (generation, idle_until)
    if close_after is not None:
        generation, idle_until = close_after
        asyncio.create_task(_close_after_reconnect_grace(port, generation, idle_until))


async def _close_after_reconnect_grace(
    port: int, generation: int, idle_until: float
) -> None:
    """Close only if no reconnect or new launch superseded this disconnect."""
    await asyncio.sleep(max(0.0, idle_until - time.monotonic()))
    async with _PORT_LOCK:
        entry = _EXTRA_PORTS.get(port)
        if (
            entry is None
            or int(entry.get("idle_generation", 0)) != generation
            or entry["viewers"] > 0
            or entry.get("leases")
        ):
            return
    await _close_port(port, only_if_idle=True)


async def _expire_lease(port: int, request_id: str, expires_at: float) -> None:
    """Expire one exact launch lease and reclaim an otherwise idle port."""
    await asyncio.sleep(max(0.0, expires_at - time.monotonic()))
    should_close = False
    async with _PORT_LOCK:
        entry = _EXTRA_PORTS.get(port)
        if entry is None:
            return
        leases = entry.setdefault("leases", {})
        if leases.get(request_id) != expires_at:
            return
        leases.pop(request_id, None)
        should_close = entry["viewers"] == 0 and not leases
    if should_close:
        _vprint(f"[ArrayView] viewer-port lease expired for {request_id[:12]}")
        await _close_port(port, only_if_idle=True)


async def _close_port(port: int, *, only_if_idle: bool = False) -> None:
    async with _PORT_LOCK:
        entry = _EXTRA_PORTS.get(port)
        if entry is None:
            return
        if only_if_idle:
            if entry["viewers"] > 0 or entry.get("leases"):
                return
            if float(entry.get("idle_until", 0.0)) > time.monotonic():
                return
        _EXTRA_PORTS.pop(port, None)
    await _shutdown_entry(port, entry)


async def _shutdown_entry(port: int, entry: dict) -> None:
    """Stop one listener already removed from the selectable port registry."""
    try:
        entry["server"].should_exit = True
        await asyncio.wait_for(entry["task"], timeout=5)
    except Exception:
        entry["task"].cancel()
    finally:
        try:
            entry["socket"].close()
        except Exception:
            pass
    _vprint(f"[ArrayView] released private viewer port {port}")


async def close_extra_ports() -> None:
    """Release every extra port.  Only safe with no viewer connected anywhere.

    A backstop for the per-port release above: if a viewer is lost without its
    disconnect being seen, this still reclaims the port once the server is idle.
    """
    if not _EXTRA_PORTS or _session_mod.VIEWER_SOCKETS > 0:
        return
    for port in list(_EXTRA_PORTS):
        await _close_port(port, only_if_idle=True)
