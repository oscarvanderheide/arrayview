import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest


@pytest.fixture
def clear_launch_env(monkeypatch):
    keys = [
        "ARRAYVIEW_WINDOW",
        "ARRAYVIEW_WINDOW_ID",
        "DISPLAY",
        "TERM_PROGRAM",
        "VSCODE_AGENT_FOLDER",
        "VSCODE_INJECTION",
        "VSCODE_IPC_HOOK_CLI",
        "WAYLAND_DISPLAY",
        "SSH_CLIENT",
        "SSH_CONNECTION",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/ping":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {
                "ok": True,
                "service": "arrayview",
                "pid": 4321,
                "hostname": "test-host",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def ping_server():
    server = HTTPServer(("localhost", 0), _PingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_launch_data_enums_match_refactor_contract():
    from arrayview._launch_plan import (
        Display,
        Environment,
        Invocation,
        Registration,
        ServerOwner,
        Transport,
    )

    assert {item.value for item in Invocation} == {
        "cli",
        "python",
        "jupyter",
        "julia",
        "matlab",
        "vscode_explorer",
        "codex",
    }
    assert {item.value for item in Environment} == {
        "terminal",
        "vscode_local",
        "vscode_remote",
        "ssh",
        "jupyter",
        "julia",
        "matlab",
    }
    assert {item.value for item in Transport} == {
        "http",
        "none",
    }
    assert {item.value for item in ServerOwner} == {
        "existing",
        "spawned_daemon",
        "in_process",
        "persistent",
        "external",
    }
    assert {item.value for item in Display} == {
        "native",
        "browser",
        "vscode",
        "inline",
        "none",
    }
    assert {item.value for item in Registration} == {
        "http_load",
        "daemon_startup",
        "in_process_session",
        "relay",
    }


def test_snapshot_classifies_terminal_and_keeps_selected_env(monkeypatch, clear_launch_env):
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import Environment, Invocation, snapshot_launch_environment

    monkeypatch.setenv("ARRAYVIEW_WINDOW", "browser")
    monkeypatch.setenv("TERM_PROGRAM", "not-vscode")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: "gtk")

    snapshot = snapshot_launch_environment(_free_port(), Invocation.CLI, "native")

    assert snapshot.invocation is Invocation.CLI
    assert snapshot.environment is Environment.TERMINAL
    assert snapshot.requested_window == "native"
    assert snapshot.env_vars == {
        "ARRAYVIEW_WINDOW": "browser",
        "DISPLAY": ":99",
        "TERM_PROGRAM": "not-vscode",
    }
    assert snapshot.config_default is None
    assert snapshot.native_backend == "gtk"
    assert snapshot.server.port_busy is False
    assert snapshot.server.arrayview_server_alive is False


def test_snapshot_reads_config_window_default(monkeypatch, clear_launch_env, tmp_path):
    import arrayview._config as config_mod
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import Environment, snapshot_launch_environment

    config_file = tmp_path / "config.toml"
    config_file.write_text('[window]\nvscode = "vscode"\ndefault = "browser"\n')
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(config_file))
    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: None)

    snapshot = snapshot_launch_environment(_free_port(), "cli")

    assert snapshot.environment is Environment.VSCODE_LOCAL
    assert snapshot.config_default == "vscode"


@pytest.mark.parametrize(
    "facts, expected",
    [
        ({"in_jupyter": True}, "jupyter"),
        ({"in_julia": True}, "julia"),
        ({"is_vscode_remote": True, "in_vscode_terminal": True}, "vscode_remote"),
        ({"in_vscode_terminal": True}, "vscode_local"),
        ({"ssh_connection": True}, "ssh"),
    ],
)
def test_classify_environment_priority(facts, expected):
    from arrayview._launch_plan import _classify_environment

    defaults = {
        "in_jupyter": False,
        "in_julia": False,
        "in_vscode_terminal": False,
        "is_vscode_remote": False,
        "ssh_connection": False,
        "ssh_client": False,
    }
    defaults.update(facts)

    assert _classify_environment(**defaults).value == expected


def test_snapshot_reports_arrayview_server_ping(monkeypatch, clear_launch_env, ping_server):
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import snapshot_launch_environment

    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: None)

    snapshot = snapshot_launch_environment(ping_server, "python")

    assert snapshot.server.port_busy is True
    assert snapshot.server.arrayview_server_alive is True
    assert snapshot.server.server_pid == 4321
    assert snapshot.server.server_hostname == "test-host"


def test_importing_launch_plan_keeps_server_and_numpy_lazy():
    code = (
        "import sys; "
        "import arrayview._launch_plan; "
        "print('arrayview._server' in sys.modules, 'numpy' in sys.modules)"
    )
    result = sys.executable

    import subprocess

    completed = subprocess.run(
        [result, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False False"


def _facts(**overrides):
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchEnvironmentSnapshot,
        ServerSnapshot,
    )

    values = dict(
        invocation=Invocation.PYTHON,
        requested_window=None,
        environment=Environment.TERMINAL,
        platform="linux",
        env_vars={},
        config_default=None,
        native_backend=None,
        server=ServerSnapshot(8123, False, False),
        in_jupyter=False,
        in_julia=False,
        in_vscode_terminal=False,
        is_vscode_remote=False,
        in_vscode_tunnel=False,
        ssh_connection=False,
        ssh_client=False,
        hostname="test-host",
    )
    values.update(overrides)
    return LaunchEnvironmentSnapshot(**values)


def test_plan_is_pure_and_json_serializable():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    facts = _facts(native_backend="gtk")
    plan = plan_launch(LaunchIntent(Invocation.PYTHON, 8123), facts)

    assert plan.ok
    assert plan.display.value == "native"
    assert plan.server_owner.value == "in_process"
    assert plan.registration.value == "in_process_session"
    assert plan.fallback_display.value == "browser"
    assert "native_available" in plan.reasons
    assert json.loads(json.dumps(plan.to_dict()))["display"] == "native"


def test_launch_context_keeps_host_placement_orthogonal_to_invocation():
    from arrayview._launch_plan import (
        CompletionTarget,
        Environment,
        Invocation,
        LaunchIntent,
        Placement,
        create_launch_context,
    )

    julia_over_ssh = _facts(
        invocation=Invocation.JULIA,
        environment=Environment.JULIA,
        in_julia=True,
        ssh_connection=True,
    )
    context = create_launch_context(
        LaunchIntent(Invocation.JULIA, 8123),
        julia_over_ssh,
        launch_id="launch-1",
    )

    assert context.launch_id == "launch-1"
    assert context.plan.environment is Environment.JULIA
    assert context.placement is Placement.SSH
    assert context.completion_target is CompletionTarget.GUIDANCE_PRINTED
    assert json.loads(json.dumps(context.to_dict()))["placement"] == "ssh"


@pytest.mark.parametrize(
    "invocation, facts, requested_window, expected_target",
    [
        ("cli", {}, "browser", "dispatch_accepted"),
        ("cli", {"native_backend": "gtk"}, "native", "frame_ready"),
        ("python", {}, "none", "session_accepted"),
        (
            "python",
            {"environment": "jupyter", "in_jupyter": True},
            "inline",
            "mime_returned_or_emitted",
        ),
        (
            "julia",
            {"environment": "julia", "in_julia": True},
            "inline",
            "display_side_effect_emitted",
        ),
    ],
)
def test_launch_context_declares_mode_specific_completion(
    invocation, facts, requested_window, expected_target
):
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchIntent,
        create_launch_context,
    )

    normalized = dict(facts)
    if "environment" in normalized:
        normalized["environment"] = Environment(normalized["environment"])
    inv = Invocation(invocation)
    context = create_launch_context(
        LaunchIntent(inv, 8123, requested_window=requested_window),
        _facts(invocation=inv, **normalized),
    )

    assert context.completion_target.value == expected_target


@pytest.mark.parametrize(
    "environment, invocation, expected_display",
    [
        ("terminal", "cli", "browser"),
        ("vscode_local", "cli", "vscode"),
        ("vscode_remote", "python", "vscode"),
        ("jupyter", "python", "inline"),
        ("julia", "julia", "browser"),
        ("ssh", "python", "browser"),
    ],
)
def test_default_display_matrix(environment, invocation, expected_display):
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchIntent,
        plan_launch,
    )

    env = Environment(environment)
    inv = Invocation(invocation)
    facts = _facts(
        invocation=inv,
        environment=env,
        in_jupyter=env is Environment.JUPYTER,
        in_julia=env is Environment.JULIA,
        in_vscode_terminal=env is Environment.VSCODE_LOCAL,
        is_vscode_remote=env is Environment.VSCODE_REMOTE,
    )
    assert plan_launch(LaunchIntent(inv, 8123), facts).display.value == expected_display


def test_explicit_native_redirects_to_vscode_in_remote():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(environment=Environment.VSCODE_REMOTE, is_vscode_remote=True)
    plan = plan_launch(LaunchIntent(Invocation.CLI, 8123, "native"), facts)

    assert plan.display.value == "vscode"
    assert plan.fallback_display is None
    assert not plan.fallback_allowed
    assert "remote_native_redirected_to_vscode" in plan.reasons


def test_explicit_browser_in_remote_vscode_stays_on_the_client_side():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(environment=Environment.VSCODE_REMOTE, is_vscode_remote=True)
    plan = plan_launch(LaunchIntent(Invocation.CLI, 8123, "browser"), facts)

    assert plan.display.value == "vscode"
    assert plan.fallback_display is None
    assert "remote_browser_redirected_to_vscode" in plan.reasons


def test_cli_browser_flag_and_config_precedence():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    configured = _facts(config_default="native", native_backend="gtk")
    explicit = plan_launch(
        LaunchIntent(Invocation.CLI, 8123, requested_window="browser", browser=True),
        configured,
    )
    flag = plan_launch(LaunchIntent(Invocation.CLI, 8123, browser=True), configured)

    assert explicit.display.value == "browser"
    assert flag.display.value == "browser"


def test_explicit_inline_false_keeps_local_native_default_and_skips_config():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    facts = _facts(config_default="inline", native_backend="gtk")
    plan = plan_launch(
        LaunchIntent(
            Invocation.PYTHON,
            8123,
            inline=False,
            inline_explicit=True,
        ),
        facts,
    )

    assert plan.display.value == "native"
    assert "config_window_default" not in plan.reasons


def test_explicit_window_false_suppresses_config_and_opens_browser():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    facts = _facts(config_default="native", native_backend="gtk")
    plan = plan_launch(
        LaunchIntent(Invocation.PYTHON, 8123, window_explicit=True),
        facts,
    )

    assert plan.display.value == "browser"
    assert "config_window_default" not in plan.reasons


def test_remote_jupyter_routes_to_vscode_unless_inline_false():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(
        environment=Environment.VSCODE_REMOTE,
        in_jupyter=True,
        in_vscode_terminal=True,
        is_vscode_remote=True,
    )

    implicit = plan_launch(LaunchIntent(Invocation.PYTHON, 8123), facts)
    explicit_inline = plan_launch(
        LaunchIntent(
            Invocation.PYTHON,
            8123,
            inline=True,
            inline_explicit=True,
        ),
        facts,
    )
    explicit_noninline = plan_launch(
        LaunchIntent(
            Invocation.PYTHON,
            8123,
            inline=False,
            inline_explicit=True,
        ),
        facts,
    )

    assert implicit.display.value == "vscode"
    assert explicit_inline.display.value == "vscode"
    assert explicit_noninline.display.value == "browser"


def test_remote_python_reuses_healthy_cli_default_server():
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchIntent,
        ServerSnapshot,
        plan_launch,
    )

    facts = _facts(
        environment=Environment.VSCODE_REMOTE,
        is_vscode_remote=True,
        cli_default_server=ServerSnapshot(8000, True, True, 42, "remote-host"),
    )
    plan = plan_launch(LaunchIntent(Invocation.PYTHON, 8123), facts)

    assert plan.effective_port == 8000
    assert plan.server_owner.value == "existing"
    assert "reuse_remote_cli_default_server" in plan.reasons


def test_local_matlab_prefers_native_when_available():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(environment=Environment.MATLAB, native_backend="gtk")
    plan = plan_launch(LaunchIntent(Invocation.MATLAB, 8123), facts)

    assert plan.display.value == "native"
    assert plan.server_owner.value == "in_process"


def test_vscode_explorer_uses_extension_owned_subprocess():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(
        environment=Environment.VSCODE_LOCAL,
        in_vscode_terminal=True,
    )
    plan = plan_launch(LaunchIntent(Invocation.VSCODE_EXPLORER, 8123), facts)

    assert plan.server_owner.value == "spawned_daemon"
    assert plan.registration.value == "daemon_startup"


def test_existing_server_is_reused_through_http():
    from arrayview._launch_plan import Invocation, LaunchIntent, ServerSnapshot, plan_launch

    facts = _facts(server=ServerSnapshot(8123, True, True, 99, "host"))
    plan = plan_launch(LaunchIntent(Invocation.CLI, 8123), facts)

    assert plan.server_owner.value == "existing"
    assert plan.registration.value == "http_load"


def test_julia_always_spawns_daemon():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(environment=Environment.JULIA, in_julia=True)
    plan = plan_launch(LaunchIntent(Invocation.JULIA, 8123), facts)

    assert plan.server_owner.value == "spawned_daemon"
    assert plan.registration.value == "daemon_startup"


def test_foreign_port_scans_locally_but_fails_in_remote():
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchFailure,
        LaunchIntent,
        ServerSnapshot,
        plan_launch,
    )

    busy = ServerSnapshot(8123, True, False)
    local = plan_launch(LaunchIntent(Invocation.CLI, 8123), _facts(server=busy))
    remote = plan_launch(
        LaunchIntent(Invocation.CLI, 8123),
        _facts(environment=Environment.VSCODE_REMOTE, server=busy),
    )

    assert local.effective_port == 8124
    assert local.ok
    assert remote.failure is LaunchFailure.REMOTE_PORT_CONFLICT
    assert not remote.ok


def test_matlab_uses_python_process_server_contract():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    plan = plan_launch(LaunchIntent(Invocation.MATLAB, 8123), _facts())

    assert plan.environment.value == "matlab"
    assert plan.server_owner.value == "in_process"
    assert plan.registration.value == "in_process_session"


def test_vscode_explorer_uses_vscode_display_contract():
    from arrayview._launch_plan import Environment, Invocation, LaunchIntent, plan_launch

    facts = _facts(
        invocation=Invocation.VSCODE_EXPLORER,
        environment=Environment.VSCODE_LOCAL,
        in_vscode_terminal=True,
    )
    plan = plan_launch(LaunchIntent(Invocation.VSCODE_EXPLORER, 8123), facts)

    assert plan.display.value == "vscode"


@pytest.mark.parametrize(
    "intent, failure",
    [
        (("python", 0, None), "invalid_port"),
        (("python", 70000, None), "invalid_port"),
        (("python", 8123, "popup"), "invalid_window"),
    ],
)
def test_typed_invalid_intent_failures(intent, failure):
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    invocation, port, window = intent
    plan = plan_launch(LaunchIntent(Invocation(invocation), port, window), _facts())

    assert plan.failure.value == failure
    assert plan.transport.value == "none"
