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
