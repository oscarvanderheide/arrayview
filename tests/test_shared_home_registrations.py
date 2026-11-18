"""Window registrations on a network home shared by several machines.

A registration written on another host carries a pid that means nothing here:
it must neither match this terminal nor be deleted as dead.
"""

import json
import os
import socket

from arrayview import _platform
from arrayview._vscode_signal import _cleanup_zombie_registrations


def _write(signal_dir, window_id, **fields):
    path = signal_dir / f"window-{window_id}.json"
    path.write_text(json.dumps({"pid": os.getpid(), "windowId": window_id, **fields}))
    return path


def test_foreign_host_registration_is_ignored_and_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    signal_dir = tmp_path / ".arrayview"
    signal_dir.mkdir()
    monkeypatch.setenv("ARRAYVIEW_WINDOW_ID", "abc")
    monkeypatch.setattr(_platform, "_current_vscode_remote_cli", lambda: None)

    # Another host's window that happens to reuse a pid alive here.
    foreign = _write(signal_dir, "abc", host=socket.gethostname() + "-other")
    assert _platform._exact_vscode_window_registration(None) is None

    # Another host's window whose pid is dead here: not ours to delete.
    dead_foreign = _write(
        signal_dir, "dead", host=socket.gethostname() + "-other", pid=2**22 + 7
    )
    _cleanup_zombie_registrations()
    assert foreign.exists() and dead_foreign.exists()

    # This host's own registration still matches.
    _write(signal_dir, "abc", host=socket.gethostname())
    match = _platform._exact_vscode_window_registration(None)
    assert match is not None and match[0] == "abc"

    # Legacy registrations without a host are judged locally, as before.
    legacy_dead = _write(signal_dir, "legacy", pid=2**22 + 7)
    _cleanup_zombie_registrations()
    assert not legacy_dead.exists()


def test_remote_vscode_skips_port_held_by_something_else():
    """A stale forward on 8000 (another user's) must not block a remote launch."""
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchEnvironmentSnapshot,
        LaunchIntent,
        ServerSnapshot,
        plan_launch,
    )

    facts = LaunchEnvironmentSnapshot(
        invocation=Invocation.CLI,
        requested_window="vscode",
        environment=Environment.VSCODE_REMOTE,
        platform="linux",
        env_vars={},
        config_default=None,
        native_backend=None,
        server=ServerSnapshot(8000, True, False),
        in_jupyter=False,
        in_julia=False,
        in_vscode_terminal=True,
        is_vscode_remote=True,
        in_vscode_tunnel=False,
        ssh_connection=True,
        ssh_client=True,
        hostname="roodnoot",
    )
    plan = plan_launch(
        LaunchIntent(Invocation.CLI, 8000, requested_window="vscode"), facts
    )
    assert plan.ok
    assert "scan_from_next_port" in plan.reasons
    assert plan.effective_port == 8001


def _remote_facts(server, own_server=None, hostname="roodnoot"):
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchEnvironmentSnapshot,
    )

    return LaunchEnvironmentSnapshot(
        invocation=Invocation.CLI,
        requested_window="vscode",
        environment=Environment.VSCODE_REMOTE,
        platform="linux",
        env_vars={},
        config_default=None,
        native_backend=None,
        server=server,
        in_jupyter=False,
        in_julia=False,
        in_vscode_terminal=True,
        is_vscode_remote=True,
        in_vscode_tunnel=False,
        ssh_connection=True,
        ssh_client=True,
        hostname=hostname,
        own_server=own_server,
    )


def _live(port, host="roodnoot", uid=None, instance="srv"):
    from arrayview._launch_plan import ServerSnapshot

    return ServerSnapshot(
        port, True, True, 4242, host, instance, None,
        ("identity-fenced-load", "identity-fenced-mutations"), "1",
        os.geteuid() if uid is None else uid,
    )


def test_same_user_server_on_another_machine_is_not_reused():
    from arrayview._launch_plan import Invocation, LaunchIntent, plan_launch

    plan = plan_launch(
        LaunchIntent(Invocation.CLI, 8000, requested_window="vscode"),
        _remote_facts(_live(8000, host="RTRSPLA06")),
    )
    assert "reuse_compatible_server" not in plan.reasons
    assert plan.effective_port == 8001


def test_own_server_on_other_port_is_reused_when_8000_is_someone_elses():
    from arrayview._launch_plan import (
        Invocation, LaunchIntent, ServerSnapshot, plan_launch,
    )

    plan = plan_launch(
        LaunchIntent(Invocation.CLI, 8000, requested_window="vscode"),
        _remote_facts(ServerSnapshot(8000, True, False), own_server=_live(8001)),
    )
    assert "reuse_own_server_on_other_port" in plan.reasons
    assert "reuse_compatible_server" in plan.reasons
    assert plan.effective_port == 8001


def test_free_requested_port_is_not_redirected_to_own_server():
    from arrayview._launch_plan import (
        Invocation, LaunchIntent, ServerSnapshot, plan_launch,
    )

    plan = plan_launch(
        LaunchIntent(Invocation.CLI, 9000, requested_window="vscode"),
        _remote_facts(ServerSnapshot(9000, False, False), own_server=_live(8001)),
    )
    assert "reuse_own_server_on_other_port" not in plan.reasons
    assert plan.effective_port == 9000


def test_relay_refuses_another_users_server_on_this_machine(monkeypatch):
    import pytest
    from arrayview import _launcher

    monkeypatch.setattr(
        _launcher,
        "_server_runtime_status",
        lambda port, host=None, **kw: {
            "instance_id": "x",
            "protocol_version": "1",
            "capabilities": ["transactional-relay-display"],
            "hostname": socket.gethostname(),
            "uid": os.geteuid() + 1,
        },
    )
    with pytest.raises(RuntimeError, match="another user"):
        _launcher._relay_array_to_server("/nonexistent.npy", 8000, "x")
