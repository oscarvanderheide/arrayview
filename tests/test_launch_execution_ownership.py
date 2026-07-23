from __future__ import annotations

import asyncio
import os
from threading import Thread as RealThread

import numpy as np
import pytest


def _install_local_python_facts(monkeypatch, launcher, launch_plan) -> None:
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher, "_in_jupyter", lambda: False)
    monkeypatch.setattr(launcher, "_is_script_mode", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(
        launcher._platform_mod,
        "_in_vscode_terminal",
        lambda: False,
    )
    monkeypatch.setattr(launcher._platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(launch_plan, "_config_window_default", lambda env: None)
    monkeypatch.setattr(launch_plan, "_native_window_gui", lambda: None)
    for key in ("SSH_CLIENT", "SSH_CONNECTION"):
        monkeypatch.delenv(key, raising=False)


def _install_immediate_in_process_server(monkeypatch, launcher):
    class ReadyEvent:
        def clear(self):
            return None

        def wait(self, timeout=None):
            return True

    thread_calls = []

    class JoinedThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            thread_calls.append({"daemon": daemon, "name": name})

        def start(self):
            worker = RealThread(target=self.target)
            worker.start()
            worker.join(timeout=2.0)
            assert not worker.is_alive()

    server_calls = []

    async def serve_background(port, **kwargs):
        server_calls.append({"port": port, **kwargs})
        await asyncio.sleep(0)

    monkeypatch.setattr(launcher, "_server_ready_event", ReadyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", JoinedThread)
    monkeypatch.setattr(launcher, "_serve_background", serve_background)
    return thread_calls, server_calls


def _capture_launch_contexts(monkeypatch, launch_plan):
    create_launch_context = launch_plan.create_launch_context
    contexts = []

    def capture(*args, **kwargs):
        context = create_launch_context(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(launch_plan, "create_launch_context", capture)
    return contexts


def test_in_process_plan_moves_to_alternate_port_instead_of_adopting_http(
    monkeypatch,
):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_local_python_facts(monkeypatch, launcher, launch_plan)
    contexts = _capture_launch_contexts(monkeypatch, launch_plan)
    planned_port = 18123
    alternate_port = planned_port + 1
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(port, False, False),
    )

    port_probes = []

    def port_in_use(port):
        port_probes.append(port)
        return port == planned_port

    monkeypatch.setattr(launcher, "_port_in_use", port_in_use)
    monkeypatch.setattr(launcher, "_server_pid", lambda port: None)
    monkeypatch.setattr(
        launcher,
        "_load_session_from_filepath",
        lambda *args, **kwargs: pytest.fail(
            "an in-process plan must never adopt a server that appeared later"
        ),
    )
    monkeypatch.setattr(launcher, "_print_viewer_location", lambda *a, **k: None)
    thread_calls, server_calls = _install_immediate_in_process_server(
        monkeypatch, launcher
    )
    monkeypatch.setattr(session_mod, "SERVER_LOOP", None)

    before_sids = set(session_mod.SESSIONS)
    try:
        handle = launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="ownership-race",
            port=planned_port,
            window=False,
        )

        assert isinstance(handle, launcher.ViewHandle)
        assert contexts[0].plan.server_owner is launch_plan.ServerOwner.IN_PROCESS
        assert (
            contexts[0].plan.registration
            is launch_plan.Registration.IN_PROCESS_SESSION
        )
        assert handle.port == alternate_port
        assert port_probes[0] == planned_port
        assert alternate_port in port_probes
        assert thread_calls == [{"daemon": True, "name": "arrayview-server"}]
        assert len(server_calls) == 1
        assert server_calls[0]["port"] == alternate_port
        assert server_calls[0]["owner_mode"] == "in_process"
        assert server_calls[0]["stop_when_closed"] is False
    finally:
        for sid in set(session_mod.SESSIONS) - before_sids:
            session_mod.SESSIONS.pop(sid, None)


@pytest.mark.parametrize(
    "current_identity",
    [None, ("replacement-instance", "replacement-process-start", 4343)],
    ids=["disappeared", "replaced-on-same-port"],
)
def test_http_load_plan_fails_closed_when_selected_server_changes(
    monkeypatch, current_identity
):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_local_python_facts(monkeypatch, launcher, launch_plan)
    contexts = _capture_launch_contexts(monkeypatch, launch_plan)
    planned_port = 18125
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(
            port,
            True,
            True,
            4242,
            "test-host",
            "planned-instance",
            "planned-process-start",
            ("identity-fenced-load", "identity-fenced-mutations"),
            "1",
            server_uid=os.geteuid(),
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: current_identity,
    )
    monkeypatch.setattr(
        launcher,
        "_load_session_from_filepath",
        lambda *args, **kwargs: pytest.fail(
            "a disappeared selected server must not receive registration"
        ),
    )
    monkeypatch.setattr(
        launcher.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail(
            "an HTTP_LOAD plan must not fall through to an in-process server"
        ),
    )

    before_sids = set(session_mod.SESSIONS)
    with pytest.raises(RuntimeError, match="selected existing ArrayView server disappeared"):
        launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="disappeared-server",
            port=planned_port,
            window=False,
        )
    assert contexts[0].plan.server_owner is launch_plan.ServerOwner.EXISTING
    assert contexts[0].plan.registration is launch_plan.Registration.HTTP_LOAD
    assert set(session_mod.SESSIONS) == before_sids


def test_display_error_after_http_registration_is_not_relabelled(monkeypatch):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    _install_local_python_facts(monkeypatch, launcher, launch_plan)
    contexts = _capture_launch_contexts(monkeypatch, launch_plan)
    planned_port = 18127
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(
            port,
            True,
            True,
            4242,
            "test-host",
            "planned-instance",
            "planned-process-start",
            ("identity-fenced-load", "identity-fenced-mutations"),
            "1",
            server_uid=os.geteuid(),
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: ("planned-instance", "planned-process-start", 4242),
    )

    registrations = []

    def register(*args, **kwargs):
        registrations.append((args, kwargs))
        return {"sid": "registered-sid"}

    class DisplayAdapterError(RuntimeError):
        pass

    def fail_display(*args, **kwargs):
        raise DisplayAdapterError("display-adapter-sentinel")

    monkeypatch.setattr(launcher, "_load_session_from_filepath", register)
    monkeypatch.setattr(launcher, "_open_browser", fail_display)
    rollbacks = []
    monkeypatch.setattr(
        launcher,
        "_release_remote_sessions",
        lambda port, sids, **kwargs: rollbacks.append((port, sids, kwargs)),
    )
    monkeypatch.setattr(
        launcher.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail(
            "successful HTTP registration must not start an in-process server"
        ),
    )

    with pytest.raises(DisplayAdapterError, match="display-adapter-sentinel"):
        launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="display-error",
            port=planned_port,
            window="browser",
        )

    assert contexts[0].plan.server_owner is launch_plan.ServerOwner.EXISTING
    assert contexts[0].plan.registration is launch_plan.Registration.HTTP_LOAD
    assert len(registrations) == 1
    assert rollbacks == [
        (
            planned_port,
            ["registered-sid"],
            {"expected_server_id": "planned-instance"},
        )
    ]
