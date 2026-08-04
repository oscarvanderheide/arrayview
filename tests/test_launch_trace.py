import importlib
import json
import os
import stat
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np


def _fresh_trace_module():
    import arrayview._launch_trace as trace

    return importlib.reload(trace)


def test_trace_is_disabled_without_an_absolute_path(monkeypatch, tmp_path):
    monkeypatch.delenv("ARRAYVIEW_LAUNCH_TRACE", raising=False)
    trace = _fresh_trace_module()

    assert trace.configure_launch_trace(path="relative.jsonl") is None
    trace.emit_launch_event("ignored")

    assert not (tmp_path / "relative.jsonl").exists()


def test_route_trace_is_inert_without_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("ARRAYVIEW_LAUNCH_TRACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    trace = _fresh_trace_module()

    trace.emit_route_launch_event(
        "page.route_entered",
        navigation_key="navigation-secret",
    )

    assert not (tmp_path / ".arrayview" / "launch-trace.jsonl").exists()


def test_route_trace_marker_enables_home_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ARRAYVIEW_LAUNCH_TRACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    trace_dir = tmp_path / ".arrayview"
    trace_dir.mkdir()
    (trace_dir / "enable-launch-trace").touch()
    trace = _fresh_trace_module()

    trace.emit_route_launch_event(
        "page.route_entered",
        navigation_key="navigation-secret",
    )

    row = json.loads((trace_dir / "launch-trace.jsonl").read_text())
    assert row["event"] == "page.route_entered"
    assert row["attrs"] == {
        "navigation_key_tag": trace.trace_tag("navigation-secret")
    }
    assert "navigation-secret" not in json.dumps(row)


def test_route_trace_explicit_environment_precedes_marker(monkeypatch, tmp_path):
    trace_dir = tmp_path / ".arrayview"
    trace_dir.mkdir()
    (trace_dir / "enable-launch-trace").touch()
    explicit_path = tmp_path / "explicit.jsonl"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ARRAYVIEW_LAUNCH_TRACE", str(explicit_path))
    trace = _fresh_trace_module()

    trace.emit_route_launch_event(
        "page.route_entered",
        navigation_key="navigation-secret",
    )

    assert explicit_path.exists()
    assert not (trace_dir / "launch-trace.jsonl").exists()


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


def test_short_route_traces_preparation_resolution_and_stale_entry(
    monkeypatch,
    tmp_path,
):
    import arrayview._session as session_mod
    from arrayview._app import app
    from arrayview._lifecycle import release_session

    monkeypatch.delenv("ARRAYVIEW_LAUNCH_TRACE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    trace_dir = tmp_path / ".arrayview"
    trace_dir.mkdir()
    (trace_dir / "enable-launch-trace").touch()
    trace = _fresh_trace_module()

    session = session_mod.Session(np.array([[1.0, 2.0], [3.0, 4.0]]))
    session_mod.SESSIONS[session.sid] = session
    phase_path = f"/viewer-phase/{session.sid}/route-trace-request"
    tab_key = "tabkey0123456789"
    first_key = "navkey0123456789"
    second_key = "navkey9876543210"

    try:
        with TestClient(app) as client:
            server_id = client.get("/ping").json()["instance_id"]

            def prepare(key, attempt, token):
                return client.post(
                    phase_path,
                    json={
                        "phase": "launch-prepared",
                        "server_id": server_id,
                        "window_id": "route-trace-window",
                        "token": token,
                        "viewer_query": f"?sid={session.sid}",
                        "tab_key": tab_key,
                        "navigation_key": key,
                        "navigation_attempt": attempt,
                    },
                )

            assert prepare(first_key, 0, "token-one").status_code == 200
            assert client.get(f"/_av/{tab_key}/{first_key}").status_code == 200
            assert prepare(second_key, 1, "token-two").status_code == 200
            assert client.get(f"/_av/{tab_key}/{first_key}").status_code == 404

        rows = [
            json.loads(line)
            for line in (trace_dir / "launch-trace.jsonl").read_text().splitlines()
        ]
        route_rows = [row for row in rows if row["event"].startswith("page.route_")]
        assert [row["event"] for row in route_rows] == [
            "page.route_prepared",
            "page.route_entered",
            "page.route_resolved",
            "page.route_retired",
            "page.route_prepared",
            "page.route_entered",
        ]
        assert route_rows[0]["attrs"]["request_id"] == "route-trace-request"
        assert route_rows[0]["attrs"]["navigation_attempt"] == 0
        assert route_rows[2]["attrs"]["request_id"] == "route-trace-request"
        assert route_rows[2]["attrs"]["navigation_attempt"] == 0
        assert route_rows[3]["attrs"]["navigation_attempt"] == 0
        assert route_rows[4]["attrs"]["navigation_attempt"] == 1
        assert route_rows[-1]["attrs"] == {
            "navigation_key_tag": trace.trace_tag(first_key),
            "tab_key_tag": trace.trace_tag(tab_key),
        }
        encoded = "\n".join(json.dumps(row) for row in route_rows)
        assert first_key not in encoded
        assert second_key not in encoded
        assert tab_key not in encoded
    finally:
        release_session(session.sid)
