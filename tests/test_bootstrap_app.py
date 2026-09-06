"""The bootstrap app answers /ping before the real server is attached and
holds every other request until it is (or fails them once loading failed)."""

import asyncio
import json
import threading
import time

import arrayview._session as session_mod
from arrayview._bootstrap_app import BootstrapApp


def _collect():
    sent = []

    async def send(message):
        sent.append(message)

    return sent, send


async def _receive_nothing():
    await asyncio.sleep(3600)


def _http_scope(path="/", method="GET"):
    return {"type": "http", "method": method, "path": path}


def _decode(sent):
    status = sent[0]["status"]
    body = b"".join(m.get("body", b"") for m in sent[1:])
    return status, json.loads(body or b"null")


def test_ping_is_answered_before_the_real_app_exists():
    app = BootstrapApp()
    sent, send = _collect()
    asyncio.run(app(_http_scope("/ping"), _receive_nothing, send))
    status, payload = _decode(sent)
    assert status == 200
    assert payload["ok"] is True
    assert payload["service"] == "arrayview"
    assert payload["instance_id"] == session_mod.SERVER_RUNTIME.instance_id
    assert payload["protocol_version"] == session_mod.SERVER_PROTOCOL_VERSION


def test_other_requests_wait_for_the_real_app_then_get_it():
    app = BootstrapApp()
    served = []

    async def real_app(scope, receive, send):
        served.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"real"})

    sent, send = _collect()

    def attach_later():
        time.sleep(0.15)
        app.attach(real_app)

    threading.Thread(target=attach_later).start()
    started = time.monotonic()
    asyncio.run(app(_http_scope("/?sid=x"), _receive_nothing, send))
    assert served == ["/?sid=x"]
    assert sent[-1]["body"] == b"real"
    assert time.monotonic() - started >= 0.14
    assert app.ready


def test_ping_goes_to_the_real_app_once_attached():
    app = BootstrapApp()
    seen = []

    async def real_app(scope, receive, send):
        seen.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    app.attach(real_app)
    sent, send = _collect()
    asyncio.run(app(_http_scope("/ping"), _receive_nothing, send))
    assert seen == ["/ping"]


def test_failed_load_returns_503_and_closes_websockets():
    app = BootstrapApp()
    app.fail(ImportError("boom"))
    sent, send = _collect()
    asyncio.run(app(_http_scope("/?sid=x"), _receive_nothing, send))
    status, payload = _decode(sent)
    assert status == 503
    assert payload["type"] == "ServerStartupFailed"
    assert "boom" in payload["error"]

    sent, send = _collect()
    asyncio.run(app({"type": "websocket", "path": "/ws"}, _receive_nothing, send))
    assert sent == [{"type": "websocket.close", "code": 1011, "reason": sent[0]["reason"]}]


def test_lifespan_is_answered_locally():
    app = BootstrapApp()
    messages = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]

    async def receive():
        return messages.pop(0)

    sent, send = _collect()
    asyncio.run(app({"type": "lifespan"}, receive, send))
    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
