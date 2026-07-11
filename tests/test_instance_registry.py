import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from arrayview._instance_registry import (
    InstanceRecord,
    InstanceRegistry,
    is_stale,
    process_start_identity,
    runtime_directory,
)


def _record(**overrides):
    values = dict(
        instance_id="11111111-1111-4111-8111-111111111111",
        pid=os.getpid(),
        process_start=process_start_identity(os.getpid()),
        port=8123,
        protocol_version="1",
        package_version="0.1",
        owner_mode="transient",
        started_at=10.0,
        last_seen_at=11.0,
        control_token="secret",
        log_path="/tmp/arrayview.log",
    )
    values.update(overrides)
    return InstanceRecord(**values)


def _hold_lock(directory, ready, release):
    registry = InstanceRegistry(directory)
    with registry.startup_lock():
        ready.set()
        release.wait(5)


def test_record_create_has_identity_and_credentials():
    record = InstanceRecord.create(
        port=9000, protocol_version="2", package_version="3",
        owner_mode="kernel", log_path="server.log",
    )
    assert record.pid == os.getpid()
    assert record.process_start == process_start_identity(os.getpid())
    assert len(record.instance_id) == 36
    assert len(record.control_token) == 64
    assert record.started_at == record.last_seen_at


def test_record_json_roundtrip_and_schema_validation():
    record = _record()
    assert InstanceRecord.from_dict(record.to_dict()) == record
    value = record.to_dict()
    value["schema"] = 999
    with pytest.raises(ValueError, match="schema"):
        InstanceRecord.from_dict(value)


def test_write_is_atomic_and_discovery_sorted(tmp_path):
    registry = InstanceRegistry(tmp_path)
    later = _record(instance_id="later", started_at=20)
    earlier = _record(instance_id="earlier", started_at=10)
    registry.write(later)
    path = registry.write(earlier)
    assert json.loads(path.read_text())["instance_id"] == "earlier"
    assert [item.instance_id for item in registry.discover()] == ["earlier", "later"]
    assert not list(registry.records.glob(".record-*"))


def test_remove_is_idempotent(tmp_path):
    registry = InstanceRegistry(tmp_path)
    registry.write(_record())
    assert registry.remove(_record().instance_id)
    assert not registry.remove(_record().instance_id)


def test_discovery_ignores_or_cleans_corrupt_and_stale_records(tmp_path, monkeypatch):
    registry = InstanceRegistry(tmp_path)
    registry.write(_record(instance_id="stale", process_start="reused-pid"))
    registry.records.joinpath("broken.json").write_text("{")
    assert registry.discover() == []
    assert len(list(registry.records.glob("*.json"))) == 2
    assert registry.discover(clean_stale=True) == []
    assert list(registry.records.glob("*.json")) == []


def test_stale_detection_rejects_pid_reuse(monkeypatch):
    monkeypatch.setattr(
        "arrayview._instance_registry.process_start_identity", lambda pid: "new-birth"
    )
    assert is_stale(_record(process_start="old-birth"))
    assert not is_stale(_record(process_start="new-birth"))


def test_process_identity_tracks_real_process_lifetime():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = process_start_identity(child.pid)
        assert identity
        assert process_start_identity(child.pid) == identity
    finally:
        child.terminate()
        child.wait(timeout=5)
    deadline = time.monotonic() + 2
    while process_start_identity(child.pid) is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process_start_identity(child.pid) is None


def test_startup_lock_excludes_another_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    try:
        with pytest.raises(TimeoutError):
            with InstanceRegistry(tmp_path).startup_lock(timeout=0.1):
                pass
    finally:
        release.set()
        process.join(5)
    assert process.exitcode == 0
    with InstanceRegistry(tmp_path).startup_lock(timeout=1):
        pass


def test_startup_lock_reaps_pid_reuse_owner(tmp_path):
    registry = InstanceRegistry(tmp_path)
    registry.lock_path.mkdir(parents=True)
    registry.lock_path.joinpath("owner.json").write_text(
        json.dumps({"pid": os.getpid(), "process_start": "not-this-process"})
    )
    with registry.startup_lock(timeout=0.2):
        assert registry.lock_path.is_dir()
    assert not registry.lock_path.exists()


def test_startup_lock_does_not_remove_replacement_owner(tmp_path):
    registry = InstanceRegistry(tmp_path)

    with registry.startup_lock(timeout=0.2):
        owner_path = registry.lock_path / "owner.json"
        replacement = {
            "pid": os.getpid(),
            "process_start": process_start_identity(os.getpid()),
            "token": "replacement-owner",
        }
        owner_path.write_text(json.dumps(replacement))

    assert registry.lock_path.is_dir()
    assert json.loads((registry.lock_path / "owner.json").read_text()) == replacement


def test_runtime_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(tmp_path / "custom"))
    assert runtime_directory() == tmp_path / "custom"


def test_runtime_directory_is_per_user_without_override(monkeypatch, tmp_path):
    monkeypatch.delenv("ARRAYVIEW_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr("arrayview._instance_registry.tempfile.gettempdir", lambda: str(tmp_path))
    first = runtime_directory()
    monkeypatch.setattr("arrayview._instance_registry.getpass.getuser", lambda: "different-user")
    second = runtime_directory()
    assert first.parent == second.parent == tmp_path
    assert first != second


def test_import_does_not_load_heavy_dependencies():
    code = "import sys; import arrayview._instance_registry; print('fastapi' in sys.modules, 'numpy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False False"
