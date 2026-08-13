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
down, and repeat opens keep using the original port.

Lifetime
--------

An extra port is closed as soon as no viewer is connected **on that port**.
Viewers report the port they arrived on, so a port is released the moment its
own viewer goes away, rather than waiting for every viewer everywhere to close —
otherwise one cold start early in a session would leave a port listening for the
rest of the day.

It must not be closed while a viewer loaded from it is alive: the viewer keeps
using HTTP on its own origin long after the page has rendered, so pulling the
port out from under it would break the viewer rather than tidy up after it.

A port that never receives a viewer at all — the launch failed, or the user
closed the tab before it loaded — is closed after ``UNUSED_GRACE_S``.  Without
that, a failed cold start would leak a port with nothing to trigger its release.
"""

from __future__ import annotations

import asyncio
import socket
import time

import arrayview._session as _session_mod
from arrayview._session import _vprint

# Cold starts are rare — one per session in practice, because the first launch
# leaves a viewer open and every launch after that is warm.  The cap is a guard
# against a pathological caller, not an expected working set.
MAX_EXTRA_PORTS = 3

# How long a freshly opened port may sit with no viewer on it before it is
# reclaimed.  This is the leak guard for a launch that never produced a viewer;
# it has to outlast a slow array load, because the port is opened before the
# page is even requested and a large file can take a while to reach its socket.
UNUSED_GRACE_S = 180.0

# port -> {"socket", "server", "task", "viewers", "opened_at"}
_EXTRA_PORTS: dict[int, dict] = {}


def extra_ports() -> list[int]:
    """Ports currently being served in addition to the main one."""
    return sorted(_EXTRA_PORTS)


def warm_port() -> int | None:
    """The extra port currently carrying a viewer, if any.

    The viewer page is served from an extra port so the main port's forwarded
    connection — which drops its first requests once it has sat idle — never
    carries a viewer.  Once one viewer is on an extra port that port is warm,
    so later launches reuse it instead of opening another: reusing the one warm
    port keeps the number of extra ports at one and keeps its forward warm for
    every later launch.
    """
    for port, entry in _EXTRA_PORTS.items():
        if entry["viewers"] > 0:
            return port
    return None


def _bind_free_port() -> socket.socket:
    """Bind a port the OS says is free, so nothing has to be guessed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sock.listen(128)
    return sock


async def open_extra_port() -> int | None:
    """Serve the running app on one more, never-before-used port.

    Returns the port, or ``None`` if one could not be added.  Never raises: a
    launch that cannot get a fresh port is still a working launch on the main
    port, only with the cold-start flicker it would have had anyway.
    """
    if len(_EXTRA_PORTS) >= MAX_EXTRA_PORTS:
        _vprint(
            f"[ArrayView] not adding another port; {len(_EXTRA_PORTS)} already open"
        )
        return None
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
        _EXTRA_PORTS[port] = {
            "socket": sock,
            "server": server,
            "task": task,
            "viewers": 0,
            "opened_at": time.monotonic(),
        }
        asyncio.create_task(_close_if_unused(port))
        # The socket is already listening and accepting before serve() runs, so
        # a client that connects immediately is queued rather than refused.
        _vprint(f"[ArrayView] serving a cold-start port on {port}")
        return port
    except Exception as exc:  # pragma: no cover - defensive
        _vprint(f"[ArrayView] could not add a cold-start port: {exc}")
        return None


def viewer_connected(port: int | None) -> None:
    """Note that a viewer arrived on *port*, if it is one of ours."""
    entry = _EXTRA_PORTS.get(port)
    if entry is not None:
        entry["viewers"] += 1


async def viewer_disconnected(port: int | None) -> None:
    """Note a viewer left *port*, and release the port if it was the last one."""
    entry = _EXTRA_PORTS.get(port)
    if entry is None:
        return
    entry["viewers"] = max(0, entry["viewers"] - 1)
    if entry["viewers"] == 0:
        await _close_port(port)


async def _close_if_unused(port: int) -> None:
    """Reclaim a port that never got a viewer.

    Its own release is driven by the viewer that loaded from it, so a launch
    that never produced one would otherwise leave the port listening forever.
    """
    await asyncio.sleep(UNUSED_GRACE_S)
    entry = _EXTRA_PORTS.get(port)
    if entry is not None and entry["viewers"] == 0:
        _vprint(f"[ArrayView] cold-start port {port} was never used")
        await _close_port(port)


async def _close_port(port: int) -> None:
    entry = _EXTRA_PORTS.pop(port, None)
    if entry is None:
        return
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
    _vprint(f"[ArrayView] released cold-start port {port}")


async def close_extra_ports() -> None:
    """Release every extra port.  Only safe with no viewer connected anywhere.

    A backstop for the per-port release above: if a viewer is lost without its
    disconnect being seen, this still reclaims the port once the server is idle.
    """
    if not _EXTRA_PORTS or _session_mod.VIEWER_SOCKETS > 0:
        return
    for port in list(_EXTRA_PORTS):
        await _close_port(port)
