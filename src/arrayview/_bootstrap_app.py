"""Answer the health check while the real server is still loading.

A freshly spawned daemon binds its port within ~150 ms, but importing the web
framework and the route modules takes another ~300-500 ms (more on a busy
host). Nothing that arrives in that window can be served by the real app, and
the launcher, the VS Code opener and the tunnel proxy all wait on ``/ping``
before they will open a viewer tab.

This tiny ASGI app is what uvicorn serves from the moment the socket is bound:

* ``GET /ping`` is answered immediately with the daemon's real identity (the
  same payload the real route returns, built from ``_session``), so the tab
  can be opened while the framework is still importing;
* every other request simply waits in line and is handed to the real app the
  moment it is attached, so the page download overlaps the import instead of
  starting after it;
* if the real app fails to load, waiting requests get a 503 and the daemon
  exits, which is what happened before too — just later.

No ordering that the launch relies on changes: ``/ping`` still reports the
identity that the registry record carries, and the viewer page still cannot
arrive before the server that serves it exists.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Awaitable, Callable

Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


class BootstrapApp:
    """ASGI app that answers ``/ping`` itself until the real app is attached."""

    def __init__(self) -> None:
        self._app = None
        self._error: BaseException | None = None
        self._ready = threading.Event()

    # ── wiring (called from whichever thread loads the real app) ──────────
    def attach(self, app) -> None:
        self._app = app
        self._ready.set()

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._ready.set()

    @property
    def ready(self) -> bool:
        return self._app is not None

    # ── ASGI ──────────────────────────────────────────────────────────────
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope.get("type")
        if kind == "lifespan":
            await self._lifespan(receive, send)
            return
        if (
            self._app is None
            and kind == "http"
            and scope.get("method") == "GET"
            and scope.get("path") == "/ping"
        ):
            await self._ping(send)
            return
        if not self._ready.is_set():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._ready.wait)
        if self._app is None:
            await self._unavailable(kind, send)
            return
        await self._app(scope, receive, send)

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        # The real app registers no startup/shutdown hooks, so uvicorn's
        # lifespan protocol is answered here and never forwarded.
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _ping(self, send: Send) -> None:
        import arrayview._session as _session_mod

        payload = _session_mod.ping_payload(
            active_sessions=len(_session_mod.SESSIONS)
        )
        await _send_json(send, 200, payload)

    async def _unavailable(self, kind: str | None, send: Send) -> None:
        detail = f"ArrayView server failed to start: {self._error!r}"
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1011, "reason": detail[:120]})
            return
        await _send_json(send, 503, {"error": detail, "type": "ServerStartupFailed"})


async def _send_json(send: Send, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
