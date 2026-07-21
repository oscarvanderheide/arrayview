import urllib.error

import pytest

from arrayview._launcher import ViewHandle


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b'{"sid":"abc","released":true}'


def test_close_posts_release_request(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return _Response()

    monkeypatch.setattr("arrayview._launcher.urllib.request.urlopen", urlopen)
    handle = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)

    handle.close()

    request, timeout = requests[0]
    assert request.full_url == "http://localhost:8123/release/abc"
    assert request.get_method() == "POST"
    assert request.data == b""
    assert timeout == 10


def test_close_fences_release_to_original_server_generation(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response()

    monkeypatch.setattr("arrayview._launcher.urllib.request.urlopen", urlopen)
    handle = ViewHandle(
        "http://localhost:8123/?sid=abc",
        "abc",
        8123,
        "server-generation-a",
    )

    handle.close()

    assert (
        requests[0].get_header("X-arrayview-expected-server-id")
        == "server-generation-a"
    )


def test_repeated_close_is_local_no_op(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response()

    monkeypatch.setattr("arrayview._launcher.urllib.request.urlopen", urlopen)
    handle = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)

    handle.close()
    handle.close()

    assert len(requests) == 1


def test_context_manager_releases_session(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return _Response()

    monkeypatch.setattr("arrayview._launcher.urllib.request.urlopen", urlopen)
    handle = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)

    with handle as entered:
        assert entered is handle

    assert len(requests) == 1


def test_close_reports_unavailable_server_and_can_retry(monkeypatch):
    attempts = 0

    def urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("arrayview._launcher.urllib.request.urlopen", urlopen)
    handle = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)

    with pytest.raises(RuntimeError, match="Failed to close viewer") as error:
        handle.close()
    with pytest.raises(RuntimeError, match="Failed to close viewer"):
        handle.close()

    assert "http://localhost:8123/release/abc" in str(error.value)
    assert attempts == 2
