import json
import sys

import pytest

from arrayview import _launcher
from arrayview._instance_registry import InstanceRecord


def _record(instance_id="instance-1", port=8123):
    return InstanceRecord(
        instance_id=instance_id,
        pid=123,
        process_start="birth",
        port=port,
        protocol_version="1",
        package_version="2.0",
        owner_mode="persistent",
        started_at=1,
        last_seen_at=2,
        control_token="top-secret-token",
        log_path="/tmp/arrayview.log",
    )


class _Registry:
    records = []

    def discover(self, *, clean_stale=False):
        assert clean_stale
        return list(self.records)


def _run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["arrayview", *args])
    _launcher.arrayview()
    return capsys.readouterr().out


def test_doctor_json_uses_plan_and_redacts_environment(monkeypatch, capsys):
    _Registry.records = [_record()]
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", "/private/socket")
    result = json.loads(
        _run(monkeypatch, capsys, "doctor", "--json", "--port", "8123")
    )
    assert result["plan"]["requested_port"] == 8123
    assert result["snapshot"]["env_vars"]["VSCODE_IPC_HOOK_CLI"] == "<redacted>"
    assert result["instances"][0]["owner_mode"] == "persistent"
    assert "control_token" not in result["instances"][0]
    assert "top-secret-token" not in json.dumps(result)


def test_instances_json_lists_public_identity(monkeypatch, capsys):
    _Registry.records = [_record("known")]
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    result = json.loads(_run(monkeypatch, capsys, "instances", "--json"))
    assert result["instances"][0]["instance_id"] == "known"
    assert "control_token" not in result["instances"][0]


def test_stop_resolves_instance_to_recorded_port(monkeypatch, capsys):
    _Registry.records = [_record("known", 9001)]
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    calls = []
    monkeypatch.setattr(
        _launcher,
        "_stop_verified_server",
        lambda port: (calls.append(port) or "stopped", 123),
    )
    output = _run(monkeypatch, capsys, "stop", "known")
    assert calls == [9001]
    assert "known: stopped" in output


def test_stop_unknown_identity_never_invokes_port_stop(monkeypatch, capsys):
    _Registry.records = [_record("known", 9001)]
    monkeypatch.setattr("arrayview._instance_registry.InstanceRegistry", _Registry)
    monkeypatch.setattr(_launcher, "_stop_verified_server", lambda _port: pytest.fail("must not stop"))
    monkeypatch.setattr(sys, "argv", ["arrayview", "stop", "unknown"])
    with pytest.raises(SystemExit):
        _launcher.arrayview()


def test_legacy_diagnose_and_kill_still_bypass_subcommands(monkeypatch):
    assert not _launcher._handle_management_command(["--diagnose"])
    assert not _launcher._handle_management_command(["--kill"])
