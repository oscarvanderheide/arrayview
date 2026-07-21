import json
import signal
from contextlib import contextmanager

import pytest

from arrayview import _launcher


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _ping(**overrides):
    payload = {
        "service": "arrayview",
        "port": 8123,
        "pid": 4321,
        "process_start": "birth-1",
        "instance_id": "matching",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        {"service": "other", "port": 8123, "pid": 4321, "process_start": "birth-1"},
        {"service": "arrayview", "port": 9999, "pid": 4321, "process_start": "birth-1"},
        {"service": "arrayview", "port": 8123},
    ],
)
def test_refuses_foreign_or_malformed_listener(monkeypatch, payload):
    signals = []
    monkeypatch.setattr(_launcher.urllib.request, "urlopen", lambda *_a, **_k: _Response(payload))
    monkeypatch.setattr(_launcher.os, "kill", lambda *args: signals.append(args))

    message, pid = _launcher._stop_verified_server(8123)

    assert pid is None
    assert "Refusing" in message
    assert signals == []


def test_refuses_pid_reuse(monkeypatch):
    signals = []
    monkeypatch.setattr(_launcher.urllib.request, "urlopen", lambda *_a, **_k: _Response(_ping()))
    monkeypatch.setattr(
        "arrayview._instance_registry.process_start_identity", lambda _pid: "birth-2"
    )
    monkeypatch.setattr(_launcher.os, "kill", lambda *args: signals.append(args))

    message, pid = _launcher._stop_verified_server(8123)

    assert pid is None
    assert "stale" in message
    assert signals == []


def test_verified_server_is_terminated_and_matching_record_removed(monkeypatch):
    signals = []
    identities = iter(["birth-1", None, None])
    removed = []

    class _Record:
        instance_id = "matching"
        port = 8123
        pid = 4321
        process_start = "birth-1"

    class _Registry:
        def discover(self, *, clean_stale):
            assert clean_stale is True
            return [_Record()]

        def remove(self, instance_id):
            removed.append(instance_id)

    monkeypatch.setattr(_launcher.urllib.request, "urlopen", lambda *_a, **_k: _Response(_ping()))
    monkeypatch.setattr(
        "arrayview._instance_registry.process_start_identity", lambda _pid: next(identities)
    )
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    monkeypatch.setattr(_launcher.os, "kill", lambda *args: signals.append(args))

    message, pid = _launcher._stop_verified_server(8123)

    assert pid == 4321
    assert "Killed process 4321" in message
    assert signals == [(4321, signal.SIGTERM)]
    assert removed == ["matching"]


def test_refuses_verified_but_unowned_arrayview_server(monkeypatch):
    signals = []

    class _Registry:
        def discover(self, *, clean_stale):
            return []

    monkeypatch.setattr(
        _launcher.urllib.request,
        "urlopen",
        lambda *_a, **_k: _Response(_ping()),
    )
    monkeypatch.setattr(
        "arrayview._instance_registry.process_start_identity",
        lambda _pid: "birth-1",
    )
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    monkeypatch.setattr(_launcher.os, "kill", lambda *args: signals.append(args))

    message, pid = _launcher._stop_verified_server(8123)

    assert pid is None
    assert "unowned" in message
    assert signals == []


def test_cli_startup_lock_rechecks_and_reuses_concurrent_server(monkeypatch):
    reused = []

    class _Registry:
        @contextmanager
        def startup_lock(self, *, timeout):
            assert timeout == 20.0
            yield

    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    monkeypatch.setattr(_launcher, "_server_alive", lambda _port: True)
    monkeypatch.setattr(
        _launcher,
        "_handle_cli_existing_server",
        lambda **kwargs: reused.append(kwargs),
    )
    monkeypatch.setattr(
        _launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not spawn a second server"),
    )
    monkeypatch.setattr(
        _launcher,
        "_configure_vscode_port_preview",
        lambda _port, **kwargs: None,
    )

    _launcher._handle_cli_spawned_daemon(
        port=8123,
        base_file="/tmp/base.npy",
        name="base.npy",
        compare_files=[],
        overlay_files=[],
        dims_override=None,
        use_native_shell=False,
        watch=False,
        window_mode="browser",
        floating=False,
        is_remote=False,
        vectorfield=None,
        vfield_components_dim=None,
        rgb=False,
        demo_name=None,
        demo_cleanup=False,
    )

    assert len(reused) == 1
    assert reused[0]["port"] == 8123
    assert reused[0]["base_file"] == "/tmp/base.npy"
