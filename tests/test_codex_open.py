from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from arrayview._launch_plan import Registration


def _context(port: int, *, existing: bool = False):
    return SimpleNamespace(
        plan=SimpleNamespace(
            ok=True,
            effective_port=port,
            failure=None,
            registration=(
                Registration.HTTP_LOAD
                if existing
                else Registration.DAEMON_STARTUP
            ),
        ),
        evidence=SimpleNamespace(
            server=SimpleNamespace(
                server_instance_id="existing-server" if existing else None
            )
        )
    )


class _Registry:
    def __init__(self, events: list[str]):
        self.events = events

    @contextmanager
    def startup_lock(self, *, timeout: float):
        self.events.append(f"lock:{timeout}")
        try:
            yield
        finally:
            self.events.append("unlock")


def test_codex_reuses_compatible_server_without_killing_or_spawning(
    monkeypatch, tmp_path
):
    import arrayview._codex_open as codex

    filepath = tmp_path / "array.npy"
    filepath.write_bytes(b"data")
    events: list[str] = []
    intents = []
    monkeypatch.setattr(
        codex,
        "create_launch_context",
        lambda intent: intents.append(intent) or _context(8000, existing=True),
    )
    monkeypatch.setattr(codex, "InstanceRegistry", lambda: _Registry(events))
    monkeypatch.setattr(
        codex,
        "_revalidate_launch_server",
        lambda context, port: port,
    )
    monkeypatch.setattr(
        codex,
        "_load_session_from_filepath",
        lambda port, path, name, **kwargs: events.append(
            f"load:{port}:{name}:{kwargs['expected_server_id']}:"
            f"release={kwargs['release_on_disconnect']}"
        )
        or {"sid": "existing-sid"},
    )
    monkeypatch.setattr(
        codex,
        "_start_loaded_server",
        lambda *args: (_ for _ in ()).throw(AssertionError("must reuse the server")),
    )
    monkeypatch.setattr(
        codex.os,
        "kill",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not kill a listener")),
    )

    url = codex._open_codex_file(str(filepath), 8000)

    assert url == "http://localhost:8000/?sid=existing-sid"
    assert events == [
        "lock:20.0",
        "unlock",
        "load:8000:array.npy:existing-server:release=True",
    ]
    assert intents[0].invocation.value == "codex"
    assert intents[0].requested_window == "browser"
    assert intents[0].persistent is True


def test_codex_uses_free_port_and_starts_under_startup_lock(monkeypatch, tmp_path):
    import arrayview._codex_open as codex

    filepath = tmp_path / "array.npy"
    filepath.write_bytes(b"data")
    events: list[str] = []
    monkeypatch.setattr(codex, "create_launch_context", lambda intent: _context(8001))
    monkeypatch.setattr(codex, "InstanceRegistry", lambda: _Registry(events))
    monkeypatch.setattr(codex, "_port_in_use", lambda port: False)

    def start(path, port, sid, name):
        assert events == ["lock:20.0"]
        events.append(f"start:{port}:{name}")
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(codex, "_start_loaded_server", start)
    monkeypatch.setattr(codex.uuid, "uuid4", lambda: SimpleNamespace(hex="new-sid"))

    url = codex._open_codex_file(str(filepath), 8000)

    assert url == "http://localhost:8001/?sid=new-sid"
    assert events == ["lock:20.0", "start:8001:array.npy", "unlock"]


def test_spawned_server_must_match_child_pid(monkeypatch):
    import arrayview._codex_open as codex

    terminated: list[bool] = []
    proc = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        terminate=lambda: terminated.append(True),
    )
    monkeypatch.setattr(codex.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(codex, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(codex, "_server_alive", lambda *args, **kwargs: True)
    monkeypatch.setattr(codex, "_server_pid", lambda port: 456)

    try:
        codex._start_loaded_server("/tmp/array.npy", 8000, "sid", "array.npy")
    except RuntimeError as exc:
        assert "claimed by another process" in str(exc)
    else:
        raise AssertionError("PID mismatch must fail")

    assert terminated == [True]


def test_codex_main_reports_actual_selected_port(monkeypatch, tmp_path, capsys):
    import arrayview._codex_open as codex

    filepath = tmp_path / "array.npy"
    filepath.write_bytes(b"data")
    monkeypatch.setattr(
        codex.sys,
        "argv",
        ["arrayview-codex", str(filepath), "--port", "8000"],
    )
    monkeypatch.setattr(
        codex,
        "_open_codex_file",
        lambda path, port: "http://localhost:8010/?sid=selected",
    )

    assert codex.main() == 0
    assert capsys.readouterr().out.strip() == "http://localhost:8010/?sid=selected"
