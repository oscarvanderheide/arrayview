"""Executable launch contracts across invocation, host, placement, and OS.

The planner tests in this module deliberately stop at the extension boundary.
``REAL_TUNNEL_ONLY_CASES`` names the scenarios that require a live VS Code
remote extension host and therefore belong in the tunnel handoff, not mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from arrayview._launch_plan import (
    Display,
    Environment,
    Invocation,
    LaunchEnvironmentSnapshot,
    LaunchFailure,
    LaunchIntent,
    Registration,
    ServerOwner,
    ServerSnapshot,
    plan_launch,
)


SUPPORTED_OSES = ("darwin", "linux", "win32")


@dataclass(frozen=True)
class ContractCase:
    name: str
    invocation: Invocation
    environment: Environment
    owner: ServerOwner
    registration: Registration
    display: Display


CONTRACT_CASES = (
    ContractCase(
        "cli_terminal", Invocation.CLI, Environment.TERMINAL,
        ServerOwner.SPAWNED_DAEMON, Registration.DAEMON_STARTUP, Display.BROWSER,
    ),
    ContractCase(
        "python_terminal", Invocation.PYTHON, Environment.TERMINAL,
        ServerOwner.IN_PROCESS, Registration.IN_PROCESS_SESSION, Display.BROWSER,
    ),
    ContractCase(
        "jupyter_kernel", Invocation.JUPYTER, Environment.JUPYTER,
        ServerOwner.IN_PROCESS, Registration.IN_PROCESS_SESSION, Display.INLINE,
    ),
    ContractCase(
        "julia_host", Invocation.JULIA, Environment.JULIA,
        ServerOwner.SPAWNED_DAEMON, Registration.DAEMON_STARTUP, Display.BROWSER,
    ),
    ContractCase(
        "matlab_host", Invocation.MATLAB, Environment.MATLAB,
        ServerOwner.IN_PROCESS, Registration.IN_PROCESS_SESSION, Display.BROWSER,
    ),
    ContractCase(
        "vscode_explorer_local", Invocation.VSCODE_EXPLORER,
        Environment.VSCODE_LOCAL, ServerOwner.SPAWNED_DAEMON,
        Registration.DAEMON_STARTUP, Display.VSCODE,
    ),
    ContractCase(
        "cli_vscode_remote", Invocation.CLI, Environment.VSCODE_REMOTE,
        ServerOwner.SPAWNED_DAEMON, Registration.DAEMON_STARTUP, Display.VSCODE,
    ),
    ContractCase(
        "python_plain_ssh", Invocation.PYTHON, Environment.SSH,
        ServerOwner.IN_PROCESS, Registration.IN_PROCESS_SESSION, Display.BROWSER,
    ),
    ContractCase(
        "codex_local", Invocation.CODEX, Environment.TERMINAL,
        ServerOwner.IN_PROCESS, Registration.IN_PROCESS_SESSION, Display.BROWSER,
    ),
)


REAL_TUNNEL_ONLY_CASES = (
    "originating_window_selected_with_multiple_tunnel_windows",
    "forwarded_url_reachable_from_local_vscode_client",
    "forwarding_privacy_and_external_uri_confirmed",
    "extension_ack_correlates_request_window_and_server_ids",
    "extension_host_restart_reconciles_existing_backend",
    "tunnel_reconnect_recovers_without_window_reload_or_kill",
)


def _facts(
    case: ContractCase,
    platform: str,
    *,
    native_backend: str | None = None,
    server: ServerSnapshot | None = None,
) -> LaunchEnvironmentSnapshot:
    environment = case.environment
    return LaunchEnvironmentSnapshot(
        invocation=case.invocation,
        requested_window=None,
        environment=environment,
        platform=platform,
        env_vars={},
        config_default=None,
        native_backend=native_backend,
        server=server or ServerSnapshot(8123, False, False),
        in_jupyter=environment is Environment.JUPYTER,
        in_julia=environment is Environment.JULIA,
        in_vscode_terminal=environment is Environment.VSCODE_LOCAL,
        is_vscode_remote=environment is Environment.VSCODE_REMOTE,
        in_vscode_tunnel=environment is Environment.VSCODE_REMOTE,
        ssh_connection=environment in {Environment.SSH, Environment.VSCODE_REMOTE},
        ssh_client=environment in {Environment.SSH, Environment.VSCODE_REMOTE},
        hostname="contract-host",
    )


@pytest.mark.parametrize("platform", SUPPORTED_OSES)
@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.name)
def test_invocation_host_placement_contract(case, platform):
    """OS facts must not change host ownership or default display policy."""
    plan = plan_launch(
        LaunchIntent(invocation=case.invocation, port=8123),
        _facts(case, platform),
    )

    assert plan.ok
    assert plan.environment is case.environment
    assert plan.server_owner is case.owner
    assert plan.registration is case.registration
    assert plan.display is case.display


@pytest.mark.parametrize("platform", SUPPORTED_OSES)
@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.name)
def test_compatible_server_reuse_is_identical_for_every_adapter(case, platform):
    """All adapters register through HTTP when a compatible server exists."""
    server = ServerSnapshot(8123, True, True, 4242, "contract-host")

    plan = plan_launch(
        LaunchIntent(invocation=case.invocation, port=8123),
        _facts(case, platform, server=server),
    )

    assert plan.ok
    assert plan.server_owner is ServerOwner.EXISTING
    assert plan.registration is Registration.HTTP_LOAD
    assert plan.effective_port == 8123


@pytest.mark.parametrize("platform", SUPPORTED_OSES)
def test_foreign_port_policy_differs_only_at_remote_forwarding_boundary(platform):
    busy = ServerSnapshot(8123, True, False)
    local_case = CONTRACT_CASES[0]
    remote_case = next(case for case in CONTRACT_CASES if case.name == "cli_vscode_remote")

    local = plan_launch(
        LaunchIntent(Invocation.CLI, 8123),
        _facts(local_case, platform, server=busy),
    )
    remote = plan_launch(
        LaunchIntent(Invocation.CLI, 8123),
        _facts(remote_case, platform, server=busy),
    )

    assert local.ok
    assert local.effective_port == 8124
    assert "scan_from_next_port" in local.reasons
    assert remote.failure is LaunchFailure.REMOTE_PORT_CONFLICT
    assert remote.effective_port == 8123


@pytest.mark.parametrize("platform", SUPPORTED_OSES)
def test_remote_native_request_preserves_vscode_fallback_contract(platform):
    remote_case = next(case for case in CONTRACT_CASES if case.name == "cli_vscode_remote")

    plan = plan_launch(
        LaunchIntent(Invocation.CLI, 8123, requested_window="native"),
        _facts(remote_case, platform, native_backend="available"),
    )

    assert plan.display is Display.VSCODE
    assert plan.fallback_display is Display.BROWSER
    assert plan.fallback_allowed
    assert "remote_native_redirected_to_vscode" in plan.reasons


def test_host_facts_override_a_generic_python_invocation():
    """Notebook and Julia detection retain priority over a generic API adapter."""
    python_case = next(case for case in CONTRACT_CASES if case.name == "python_terminal")
    base = _facts(python_case, "linux")

    jupyter = LaunchEnvironmentSnapshot(
        **{**base.__dict__, "environment": Environment.TERMINAL, "in_jupyter": True}
    )
    julia = LaunchEnvironmentSnapshot(
        **{**base.__dict__, "environment": Environment.TERMINAL, "in_julia": True}
    )

    jupyter_plan = plan_launch(LaunchIntent(Invocation.PYTHON, 8123), jupyter)
    julia_plan = plan_launch(LaunchIntent(Invocation.PYTHON, 8123), julia)

    assert (jupyter_plan.environment, jupyter_plan.display) == (
        Environment.JUPYTER,
        Display.INLINE,
    )
    assert (julia_plan.environment, julia_plan.server_owner) == (
        Environment.JULIA,
        ServerOwner.SPAWNED_DAEMON,
    )


def test_real_tunnel_only_cases_are_explicit_and_not_claimed_as_local_coverage():
    """Keep the live extension/forwarding boundary visible in test reporting."""
    assert REAL_TUNNEL_ONLY_CASES == (
        "originating_window_selected_with_multiple_tunnel_windows",
        "forwarded_url_reachable_from_local_vscode_client",
        "forwarding_privacy_and_external_uri_confirmed",
        "extension_ack_correlates_request_window_and_server_ids",
        "extension_host_restart_reconciles_existing_backend",
        "tunnel_reconnect_recovers_without_window_reload_or_kill",
    )
