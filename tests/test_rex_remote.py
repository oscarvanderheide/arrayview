"""A Rex terminal on a remote machine prints a link instead of opening a window."""

import json
import subprocess
import types

import pytest

from arrayview import _platform
from arrayview._launch_plan import (
    CompletionTarget,
    Display,
    Environment,
    Invocation,
    LaunchEnvironmentSnapshot,
    LaunchIntent,
    ServerSnapshot,
    _completion_target,
    _placement,
    Placement,
    plan_launch,
)


def _clients(app_pid: int, transport: str = "unix") -> str:
    return json.dumps(
        {
            "clients": [
                {
                    "info": {"kind": "app", "platform": "macos"},
                    "principal": {"transport": transport, "pid": app_pid},
                    "connection_state": "connected",
                },
                {
                    "info": {"kind": "cli", "platform": "linux"},
                    "principal": {"transport": "unix", "pid": 1},
                    "connection_state": "connected",
                },
            ]
        }
    )


@pytest.fixture
def rex_env(monkeypatch, tmp_path):
    rex = tmp_path / "rex"
    rex.write_text("")
    monkeypatch.setenv("TERM_PROGRAM", "rex")
    monkeypatch.setenv("REX_SERVER", "unix:///tmp/rex.sock")
    monkeypatch.setenv("REX_BIN_DIR", str(tmp_path))
    monkeypatch.setattr(_platform, "_rex_remote_cache", None)

    def install(clients_json: str, process_name: str):
        def fake_run(cmd, **kwargs):
            out = clients_json if cmd[0].endswith("rex") else process_name + "\n"
            return types.SimpleNamespace(stdout=out, returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

    return install


def test_app_reaching_server_through_sshd_is_remote(rex_env):
    rex_env(_clients(4242), "sshd")
    assert _platform._in_rex_remote() is True


def test_remote_without_rex_server_variable(rex_env, monkeypatch):
    # A systemd socket-activated Rex server does not set REX_SERVER.
    monkeypatch.delenv("REX_SERVER")
    rex_env(_clients(4242), "sshd")
    assert _platform._in_rex_remote() is True


def test_app_over_network_is_remote(rex_env):
    rex_env(_clients(4242, transport="tcp"), "anything")
    assert _platform._in_rex_remote() is True


def test_app_on_this_machine_is_local(rex_env):
    rex_env(_clients(4242), "/Applications/Rex.app/Contents/MacOS/Rex")
    assert _platform._in_rex_remote() is False


def test_not_rex_terminal(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.setattr(_platform, "_rex_remote_cache", None)
    assert _platform._in_rex_remote() is False


def test_rex_remote_plans_printed_link():
    facts = LaunchEnvironmentSnapshot(
        invocation=Invocation.CLI,
        requested_window=None,
        environment=Environment.SSH,
        platform="linux",
        env_vars={},
        config_default=None,
        native_backend="",
        server=ServerSnapshot(8000, False, False),
        in_jupyter=False,
        in_julia=False,
        in_vscode_terminal=False,
        is_vscode_remote=False,
        in_vscode_tunnel=False,
        ssh_connection=False,
        ssh_client=False,
        hostname="remote",
        rex_remote=True,
    )
    plan = plan_launch(LaunchIntent(invocation=Invocation.CLI, port=8000), facts)
    assert plan.display is Display.BROWSER
    assert _placement(facts) is Placement.SSH
    assert _completion_target(Invocation.CLI, plan, facts) is (
        CompletionTarget.GUIDANCE_PRINTED
    )


def _status_event(record: dict) -> dict:
    return {
        "type": "session_event",
        "payload": {
            "session_id": "session:1",
            "event": {
                "type": "block_event",
                "payload": {
                    "block_id": "block:1",
                    "name": "program_status_changed",
                    "payload": {"id": record.get("id"), "record": record},
                },
            },
        },
    }


def test_mac_helper_reads_arrayview_report():
    from arrayview._rex import _viewer_url_from_event

    url = "http://localhost:8000/?sid=abc"
    event = _status_event({"id": "arrayview", "app": "arrayview", "msg": url})
    assert _viewer_url_from_event(event) == url


def test_mac_helper_ignores_other_programs_and_hosts():
    from arrayview._rex import _viewer_url_from_event

    other = _status_event({"id": "build", "app": "make", "msg": "http://localhost:1/"})
    remote_host = _status_event(
        {"id": "arrayview", "app": "arrayview", "msg": "http://evil.example:80/"}
    )
    assert _viewer_url_from_event(other) is None
    assert _viewer_url_from_event(remote_host) is None
    assert _viewer_url_from_event({"type": "client_joined"}) is None


def test_announce_writes_status_report(monkeypatch, tmp_path):
    import base64
    import builtins

    from arrayview import _rex

    tty = tmp_path / "tty"
    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: real_open(tty if path == "/dev/tty" else path, *a, **k),
    )
    assert _rex.announce("http://localhost:8000/?sid=abc")
    text = tty.read_text()
    assert text.startswith("\033]7501;state=idle:app=arrayview:id=arrayview")
    msg = text.split("msg=")[1].split("\033")[0]
    import json

    data = json.loads(base64.b64decode(msg + "=" * (-len(msg) % 4)).decode())
    assert data["url"] == "http://localhost:8000/?sid=abc"
    assert data["view"] == "native"


def test_browser_request_reaches_mac_helper():
    from arrayview._rex import _report_from_event

    url = "http://localhost:8000/?sid=abc"
    import json

    browser = _status_event(
        {"id": "arrayview", "app": "arrayview",
         "msg": json.dumps({"url": url, "view": "browser"})}
    )
    default = _status_event({"id": "arrayview", "app": "arrayview", "msg": url})
    assert _report_from_event(browser) == (url, "browser")
    assert _report_from_event(default) == (url, "native")


def test_window_browser_plans_explicit_browser():
    facts = LaunchEnvironmentSnapshot(
        invocation=Invocation.CLI, requested_window="browser",
        environment=Environment.SSH, platform="linux", env_vars={},
        config_default=None, native_backend=None,
        server=ServerSnapshot(8000, False, False), in_jupyter=False,
        in_julia=False, in_vscode_terminal=False, is_vscode_remote=False,
        in_vscode_tunnel=False, ssh_connection=False, ssh_client=True,
        hostname="remote", rex_remote=True,
    )
    plan = plan_launch(
        LaunchIntent(invocation=Invocation.CLI, port=8000,
                     requested_window="browser", window_explicit=True),
        facts,
    )
    assert "explicit_browser" in plan.reasons
