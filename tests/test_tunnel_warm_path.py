"""Real HTTP regression coverage for stalled tunnel warming requests."""

import threading
import time
from urllib.parse import parse_qs

import pytest

from arrayview._app import app


pytestmark = pytest.mark.browser


class StalledWarmEndpoint:
    """Hold real HTTP requests until Chromium closes their connections."""

    def __init__(self, downstream, after_headers=False):
        self.downstream = downstream
        self.after_headers = after_headers
        self.lock = threading.Lock()
        self.hold = True
        self.active = 0
        self.maximum = 0
        self.started = 0
        self.disconnected = 0
        self.completed = 0

    def counts(self):
        with self.lock:
            return self.active, self.maximum, self.started, self.disconnected, self.completed

    async def __call__(self, scope, receive, send):
        warm = (
            scope["type"] == "http"
            and scope["path"] == "/ping"
            and "warm" in parse_qs(scope["query_string"].decode())
        )
        if not warm:
            return await self.downstream(scope, receive, send)
        with self.lock:
            self.started += 1
            held = self.hold
            if held:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
        if not held:
            await self.downstream(scope, receive, send)
            with self.lock:
                self.completed += 1
            return
        try:
            if self.after_headers:
                await send({
                    "type": "http.response.start", "status": 200,
                    "headers": [(b"content-length", b"2")],
                })
                await send({
                    "type": "http.response.body", "body": b"{", "more_body": True,
                })
            while (await receive())["type"] != "http.disconnect":
                pass
            with self.lock:
                self.disconnected += 1
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture(params=[False, True], ids=["before-headers", "during-body"])
def warm_endpoint(server_url, context, monkeypatch, request):
    # Wrap the running ASGI stack rather than intercepting Playwright requests:
    # these requests must consume Chromium's real per-origin HTTP connections.
    endpoint = StalledWarmEndpoint(app.middleware_stack, request.param)
    monkeypatch.setattr(app, "middleware_stack", endpoint)
    yield endpoint
    context.close()


def wait_counts(page, endpoint, predicate, timeout=5.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        counts = endpoint.counts()
        if predicate(counts):
            return counts
        page.wait_for_timeout(50)
    pytest.fail(f"Warm request counts did not recover: {endpoint.counts()}")


def open_viewer(page, server_url, sid, *, integrated=True):
    url = server_url.replace("127.0.0.1", "localhost")
    query = "&_av_integrated_browser=1" if integrated else ""
    page.goto(f"{url}/?sid={sid}{query}", wait_until="domcontentloaded", timeout=5_000)
    page.wait_for_function("() => lastImageData !== null", timeout=5_000)


def test_repeated_warming_does_not_accumulate_and_disconnects(
    page, server_url, sid_2d, warm_endpoint
):
    open_viewer(page, server_url, sid_2d)
    page.evaluate("_warmClientPath(3)")
    wait_counts(page, warm_endpoint, lambda c: c[0] == 3)
    page.evaluate("_warmClientPath(3); _warmClientPath(3)")
    page.wait_for_timeout(300)
    assert warm_endpoint.counts()[1] == 3, "Repeated warming occupied more HTTP connections"
    counts = wait_counts(page, warm_endpoint, lambda c: c[0] == 0 and c[3] >= 3)
    assert counts[2] == 3, "Overlapping warming queued another batch"


def test_two_stalled_viewers_allow_a_fresh_tab_to_render(
    page, context, server_url, sid_2d, sid_3d, warm_endpoint
):
    open_viewer(page, server_url, sid_2d)
    second = context.new_page()
    open_viewer(second, server_url, sid_3d)
    page.evaluate("_warmClientPath(3)")
    second.evaluate("_warmClientPath(3)")
    wait_counts(page, warm_endpoint, lambda c: c[0] == 6)
    third = context.new_page()
    # A real third navigation queues behind the six held HTTP connections.
    # Client cancellation must free them, without closing either old viewer.
    open_viewer(third, server_url, sid_2d)
    counts = wait_counts(third, warm_endpoint, lambda c: c[0] == 0 and c[3] >= 6)
    assert counts[1] == 6
    assert page.evaluate("lastImageData !== null")
    assert second.evaluate("lastImageData !== null")


def test_failed_page_stops_warming_but_foreground_and_new_viewer_recover(
    page, context, server_url, sid_2d, warm_endpoint
):
    open_viewer(page, server_url, sid_2d)
    page.evaluate("_warmClientPath(3)")
    wait_counts(page, warm_endpoint, lambda c: c[0] == 3)
    wait_counts(page, warm_endpoint, lambda c: c[0] == 0 and c[3] == 3)
    with warm_endpoint.lock:
        warm_endpoint.hold = False
    before = warm_endpoint.counts()[2]
    page.evaluate("_warmClientPath(3)")
    page.wait_for_timeout(300)
    assert warm_endpoint.counts()[2] == before, "Failed page restarted background warming"
    assert page.evaluate("async () => (await fetch('/ping')).ok")
    fresh = context.new_page()
    open_viewer(fresh, server_url, sid_2d)
    fresh.evaluate("_warmClientPath(3)")
    wait_counts(fresh, warm_endpoint, lambda c: c[4] == 3)
    fresh.wait_for_function(
        "() => performance.getEntriesByType('resource').filter("
        "entry => entry.name.includes('/ping?warm=') && entry.responseEnd > 0"
        ").length === 3",
        timeout=5_000,
    )
    fresh.evaluate("_warmClientPath(3)")
    wait_counts(fresh, warm_endpoint, lambda c: c[4] == 6)
    normal = context.new_page()
    open_viewer(normal, server_url, sid_2d, integrated=False)
    before = warm_endpoint.counts()[2]
    normal.evaluate("_warmClientPath(6)")
    normal.wait_for_timeout(300)
    assert warm_endpoint.counts()[2] == before
