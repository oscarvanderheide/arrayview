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

An extra port is closed once no viewer is connected anywhere, which is both the
moment it cannot be in use and the moment it is no longer needed.  It must not
be closed while a viewer loaded from it is alive: the viewer keeps using HTTP on
its own origin after the page loads, so pulling the port out from under it would
break the viewer rather than tidy up after it.
"""

from __future__ import annotations

import asyncio
import socket

import arrayview._session as _session_mod
from arrayview._session import _vprint

# Cold starts are rare — one per session in practice, because the first launch
# leaves a viewer open and every launch after that is warm.  The cap is a guard
# against a pathological caller, not an expected working set.
MAX_EXTRA_PORTS = 3

# port -> {"socket", "server", "task"}
_EXTRA_PORTS: dict[int, dict] = {}


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
        _EXTRA_PORTS[port] = {"socket": sock, "server": server, "task": task}
        # The socket is already listening and accepting before serve() runs, so
        # a client that connects immediately is queued rather than refused.
        _vprint(f"[ArrayView] serving a cold-start port on {port}")
        return port
    except Exception as exc:  # pragma: no cover - defensive
        _vprint(f"[ArrayView] could not add a cold-start port: {exc}")
        return None


async def close_extra_ports() -> None:
    """Release every extra port.

    Only safe when no viewer is connected: a viewer keeps making HTTP requests
    to the origin it was loaded from long after its page has rendered, so
    closing that port underneath it breaks the viewer.
    """
    if not _EXTRA_PORTS:
        return
    if _session_mod.VIEWER_SOCKETS > 0:
        return
    for port, entry in list(_EXTRA_PORTS.items()):
        _EXTRA_PORTS.pop(port, None)
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
