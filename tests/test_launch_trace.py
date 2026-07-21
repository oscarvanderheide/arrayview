import importlib
import json
import os
import stat
import subprocess
import sys


def _fresh_trace_module():
    import arrayview._launch_trace as trace

    return importlib.reload(trace)


def test_trace_is_disabled_without_an_absolute_path(monkeypatch, tmp_path):
    monkeypatch.delenv("ARRAYVIEW_LAUNCH_TRACE", raising=False)
    trace = _fresh_trace_module()

    assert trace.configure_launch_trace(path="relative.jsonl") is None
    trace.emit_launch_event("ignored")

    assert not (tmp_path / "relative.jsonl").exists()


def test_trace_writes_private_jsonl_with_stable_schema(monkeypatch, tmp_path):
    path = tmp_path / "launch.jsonl"
    monkeypatch.setenv("ARRAYVIEW_LAUNCH_TRACE", str(path))
    trace = _fresh_trace_module()

    assert trace.configure_launch_trace(launch_id="launch-1", role="parent") == "launch-1"
    trace.emit_launch_event("plan.selected", primary_display="native")

    row = json.loads(path.read_text())
    assert row["schema"] == 1
    assert row["launch_id"] == "launch-1"
    assert row["role"] == "parent"
    assert row["seq"] == 1
    assert row["event"] == "plan.selected"
    assert row["attrs"] == {"primary_display": "native"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_child_environment_is_copied_not_mutated(monkeypatch, tmp_path):
    path = tmp_path / "launch.jsonl"
    monkeypatch.setenv("ARRAYVIEW_LAUNCH_TRACE", str(path))
    monkeypatch.delenv("ARRAYVIEW_LAUNCH_ID", raising=False)
    trace = _fresh_trace_module()
    trace.configure_launch_trace(launch_id="launch-2", role="parent")
    original = {"PATH": "/test/bin"}

    child = trace.trace_child_environment(original)

    assert original == {"PATH": "/test/bin"}
    assert child == {
        "PATH": "/test/bin",
        "ARRAYVIEW_LAUNCH_TRACE": str(path),
        "ARRAYVIEW_LAUNCH_ID": "launch-2",
        "ARRAYVIEW_LAUNCH_ROLE": "daemon",
    }
    assert "ARRAYVIEW_LAUNCH_ID" not in os.environ


def test_trace_failure_cannot_escape(monkeypatch, tmp_path):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    monkeypatch.setenv("ARRAYVIEW_LAUNCH_TRACE", str(directory))
    trace = _fresh_trace_module()
    trace.configure_launch_trace(launch_id="launch-3")

    trace.emit_launch_event("bad-path")
    trace.emit_launch_event("bad-json", value=object())


def test_two_processes_append_parseable_events(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    code = (
        "from arrayview._launch_trace import emit_launch_event;"
        "[emit_launch_event('worker.event', value=i) for i in range(20)]"
    )
    base_env = dict(os.environ)
    base_env["ARRAYVIEW_LAUNCH_TRACE"] = str(path)
    base_env["ARRAYVIEW_LAUNCH_ID"] = "shared-launch"
    processes = []
    for role in ("worker-a", "worker-b"):
        env = dict(base_env)
        env["ARRAYVIEW_LAUNCH_ROLE"] = role
        processes.append(subprocess.Popen([sys.executable, "-c", code], env=env))

    for process in processes:
        assert process.wait(timeout=10) == 0

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 40
    assert {row["launch_id"] for row in rows} == {"shared-launch"}
    assert {row["role"] for row in rows} == {"worker-a", "worker-b"}
    for role in ("worker-a", "worker-b"):
        assert sorted(row["seq"] for row in rows if row["role"] == role) == list(
            range(1, 21)
        )
