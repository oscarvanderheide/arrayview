import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

import arrayview._source_safety as safety


def _mountinfo(mountpoint: Path, *, filesystem: str = "cifs") -> str:
    escaped = str(mountpoint).replace(" ", r"\040")
    return (
        f"42 31 0:55 / {escaped} rw,relatime - {filesystem} "
        "//server/share rw,addr=10.20.30.40,soft\n"
    )


def _debug_data(*, disconnected: bool) -> str:
    marker = " DISCONNECTED [RECONNECTING]" if disconnected else ""
    status = 3 if disconnected else 1
    return (
        "Display Internal CIFS Data Structures for Debugging\n"
        "1) ConnectionId: 0x1 Hostname: server\n"
        f"TCP status: {status}\n"
        f"SMB session status: {status} Address: 10.20.30.40\n"
        f"Shares:\n1) \\\\server\\share Status: {status}{marker}\n"
    )


def _proc_fixtures(monkeypatch, tmp_path, mountpoint, *, disconnected=False):
    mountinfo = tmp_path / "mountinfo"
    debug = tmp_path / "DebugData"
    mountinfo.write_text(_mountinfo(mountpoint))
    debug.write_text(_debug_data(disconnected=disconnected))
    monkeypatch.setenv("_ARRAYVIEW_TEST_MOUNTINFO", str(mountinfo))
    monkeypatch.setenv("_ARRAYVIEW_TEST_CIFS_DEBUG", str(debug))
    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv(
        "_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT", str(tmp_path / "source-safety")
    )
    # Staging is opt-in by default now; these tests exercise the staging
    # subsystem itself, so turn it on regardless of the ambient default.
    monkeypatch.setenv("ARRAYVIEW_SKIP_SOURCE_STAGING", "0")


def test_mountinfo_uses_nested_component_boundary_and_unescapes():
    records = safety.parse_mountinfo(
        "10 1 0:1 / /smb rw - cifs //server/root rw,addr=1.2.3.4\n"
        "11 10 0:2 / /smb/nested\\040share rw - cifs //server/nested rw,addr=5.6.7.8\n"
    )
    assert safety.mount_for_path("/smb/nested share/file.nii", records).mount_id == 11
    assert safety.mount_for_path("/smb/nested-share/file.nii", records).mount_id == 10


def test_stacked_mount_selects_visible_child():
    records = safety.parse_mountinfo(
        "10 1 0:1 / /mnt/data rw - ext4 /dev/a rw\n"
        "11 10 0:2 / /mnt/data rw - cifs //server/share rw,addr=1.2.3.4\n"
    )
    assert safety.mount_for_path("/mnt/data/file.nii", records).mount_id == 11


def test_recursive_scan_detects_nested_network_mount(monkeypatch):
    records = safety.parse_mountinfo(
        "10 1 0:1 / /data rw - ext4 /dev/a rw\n"
        "11 10 0:2 / /data/nested rw - cifs //server/share rw,addr=1.2.3.4\n"
    )
    monkeypatch.setattr(safety, "read_mountinfo", lambda: records)
    assert safety.network_mount_below("/data").mount_id == 11


def test_known_disconnected_cifs_never_starts_target_access(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=True)
    touched = []
    monkeypatch.setattr(
        safety,
        "_stage_network_file",
        lambda *args, **kwargs: touched.append(args) or pytest.fail("target touched"),
    )

    with pytest.raises(safety.UnsafeSourceError, match="disconnected"):
        safety.prepare_source(str(mountpoint / "blocked.nii"))
    assert touched == []


@pytest.mark.parametrize("dfs", [False, True], ids=["direct", "dfs"])
def test_disconnected_cifs_tcon_uses_multiline_share_stanza(dfs):
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    identity = (
        "DFS origin fullpath: \\\\server\\share\\nested\n"
        if dfs
        else "1) \\\\server\\share\n"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\n"
        "TCP status: 1\n"
        "SMB session status: 1 Address: 10.20.30.40\n"
        "Shares:\n"
        + ("1) \\\\server\\other\n" if dfs else "")
        + identity
        + "PathComponentMax: 255 Status: 3 type: DISK DISCONNECTED\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "disconnected"


def test_exact_disconnected_tcon_wins_over_healthy_same_address_sibling():
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\nTCP status: 1\n"
        "SMB session status: 1 Address: 10.20.30.40\nShares:\n"
        "1) \\\\server\\share\nPathComponentMax: 255 Status: 3 DISCONNECTED\n"
        "2) ConnectionId: 0x2 Hostname: server\nTCP status: 1\n"
        "SMB session status: 1 Address: 10.20.30.40\nShares:\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "disconnected"


def test_reconnecting_tcp_is_disconnected_before_session_status_changes():
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\nTCP status: 3\n"
        "SMB session status: 1 Address: 10.20.30.40\nShares:\n"
        "1) \\\\server\\share Status: 1\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "disconnected"


def test_unhealthy_extra_channel_does_not_override_healthy_primary():
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\nTCP status: 1\n"
        "SMB session status: 1 Address: 10.20.30.40\nShares:\n"
        "1) \\\\server\\share Status: 1\nExtra Channels:\nTCP status: 3\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "healthy"


def test_exact_share_match_prefers_mount_address():
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\nTCP status: 1\n"
        "Address: 10.20.30.40 Session Status: 1\nShares:\n"
        "1) \\\\server\\share Status: 1\n"
        "2) ConnectionId: 0x2 Hostname: server\nTCP status: 3\n"
        "Name: 10.20.30.41 SMB session status: 3\nShares:\n"
        "1) \\\\server\\share Status: 3 DISCONNECTED\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "healthy"


def test_exact_share_match_detects_bad_mount_address_among_same_unc():
    mount = safety.MountRecord(
        42, 31, "/smb/share", "cifs", "//server/share", "rw,addr=10.20.30.40"
    )
    debug = (
        "1) ConnectionId: 0x1 Hostname: server\nTCP status: 3\n"
        "Name: 10.20.30.40 SMB session status: 3\nShares:\n"
        "1) \\\\server\\share Status: 3 DISCONNECTED\n"
        "2) ConnectionId: 0x2 Hostname: server\nTCP status: 1\n"
        "Address: 10.20.30.41 Session Status: 1\nShares:\n"
        "1) \\\\server\\share Status: 1\n"
    )
    assert safety.cifs_connection_state(mount, debug) == "disconnected"


def test_wildcard_scan_root_is_complete_parent():
    assert safety.scan_root_before_magic("/data/case*/**/*.nii") == "/data"


def test_safe_staging_root_rejects_preexisting_symlink(monkeypatch, tmp_path):
    monkeypatch.delenv("_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    suffix = str(os.getuid()) if hasattr(os, "getuid") else str(os.getpid())
    root = tmp_path / f"arrayview-source-{suffix}"
    root.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    records = safety.parse_mountinfo(
        f"10 1 0:1 / {tmp_path} rw - tmpfs tmpfs rw\n"
    )
    monkeypatch.setattr(safety, "read_mountinfo", lambda: records)

    with pytest.raises(safety.UnsafeSourceError, match="unsafe or unavailable"):
        safety._safe_storage_root()


def test_private_staging_child_rejects_preexisting_symlink(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "source.nii"
    source.write_bytes(b"not opened")
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    root = Path(os.environ["_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT"])
    root.mkdir()
    (root / "staging").symlink_to(tmp_path / "elsewhere", target_is_directory=True)

    with pytest.raises(safety.UnsafeSourceError, match="private directory"):
        safety.prepare_source(str(source), timeout=0.1)
    assert not list((root / "mount-guards").glob("*"))


def test_healthy_network_nifti_is_staged_and_loads(monkeypatch, tmp_path):
    nib = pytest.importorskip("nibabel")
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "healthy.nii"
    expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    nib.save(nib.Nifti1Image(expected, np.eye(4)), source)
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)

    prepared = safety.prepare_source(str(source), timeout=5)
    try:
        assert prepared.is_network
        assert prepared.launch_path != str(source)
        from arrayview._io import load_data_with_meta

        data, metadata = load_data_with_meta(prepared.launch_path)
        np.testing.assert_array_equal(np.asarray(data), expected)
        assert metadata is not None
    finally:
        safety.cleanup_prepared_sources([prepared])


def test_network_container_that_can_reference_other_paths_is_rejected(
    monkeypatch, tmp_path
):
    mountpoint = tmp_path / "share"
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    monkeypatch.setattr(
        safety,
        "_stage_network_file",
        lambda *args, **kwargs: pytest.fail("unsafe container was staged"),
    )
    with pytest.raises(safety.UnsafeSourceError, match="reference other files"):
        safety.prepare_source(str(mountpoint / "external-links.h5"))


def test_indefinitely_blocked_helper_is_bounded_reaped_and_quarantined(
    monkeypatch, tmp_path
):
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "source.nii"
    source.write_bytes(b"not read by the simulated helper")
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    real_popen = subprocess.Popen
    children = []

    def sleeping_popen(*args, **kwargs):
        child = real_popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            stdin=kwargs.get("stdin"),
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
            close_fds=True,
            start_new_session=True,
        )
        children.append(child)
        return child

    monkeypatch.setattr(safety.subprocess, "Popen", sleeping_popen)
    started = time.monotonic()
    with pytest.raises(safety.UnsafeSourceError, match="Timed out"):
        safety.prepare_source(str(source), timeout=0.1)
    assert time.monotonic() - started < 3
    assert children[0].poll() is not None

    with pytest.raises(safety.UnsafeSourceError, match="quarantined"):
        safety.prepare_source(str(source), timeout=0.1)
    assert len(children) == 1


def test_unreaped_helper_keeps_persistent_quarantine(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "source.nii"
    source.write_bytes(b"source")
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)

    class UnreapedProcess:
        pid = os.getpid()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("helper", timeout)

        def terminate(self):
            pass

        def kill(self):
            pass

    started = []
    monkeypatch.setattr(
        safety.subprocess,
        "Popen",
        lambda *args, **kwargs: started.append(UnreapedProcess()) or started[-1],
    )
    monkeypatch.setattr(safety, "_stop_helper", lambda process: False)

    with pytest.raises(safety.UnsafeSourceError, match="could not be reaped"):
        safety.prepare_source(str(source), timeout=0.1)
    with pytest.raises(safety.UnsafeSourceError, match="quarantined"):
        safety.prepare_source(str(source), timeout=0.1)
    assert len(started) == 1


def test_helper_spawn_failure_cleans_guard_and_staging(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "source.nii"
    source.write_bytes(b"source")
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    monkeypatch.setattr(
        safety.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(safety.UnsafeSourceError, match="spawn failed"):
        safety.prepare_source(str(source), timeout=0.1)
    root = Path(os.environ["_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT"])
    assert not list((root / "mount-guards").glob("*"))
    assert not list((root / "staging").glob("*"))


def test_unreaped_guard_recovers_after_exact_helper_is_gone(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    mount = safety.mount_for_path(str(mountpoint / "source.nii"))
    guard = safety._quarantine_path(mount)
    guard.mkdir(parents=True)
    (guard / "state.json").write_text(
        '{"pid": 99999999, "process_start": "old", "status": "unreaped", '
        '"updated_at": 0}'
    )

    safety._check_quarantine(mount)
    assert not guard.exists()


def test_stale_guard_recovery_removes_its_owned_staging(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    mount = safety.mount_for_path(str(mountpoint / "source.nii"))
    root = Path(os.environ["_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT"])
    staging = root / "staging" / "orphaned"
    staging.mkdir(parents=True)
    (staging / "source.nii").write_bytes(b"snapshot")
    guard = safety._quarantine_path(mount)
    guard.mkdir(parents=True)
    (guard / "state.json").write_text(
        json.dumps(
            {
                "pid": 99999999,
                "process_start": "old",
                "status": "in_progress",
                "updated_at": 0,
                "staging_dir": str(staging),
            }
        )
    )

    safety._check_quarantine(mount)
    assert not staging.exists()
    assert not guard.exists()


def test_stop_helper_tolerates_process_exit_during_terminate():
    class ExitedProcess:
        def terminate(self):
            raise ProcessLookupError()

        def wait(self, timeout=None):
            return 0

    assert safety._stop_helper(ExitedProcess())


def test_initial_guard_state_failure_removes_guard(monkeypatch, tmp_path):
    mountpoint = tmp_path / "share"
    _proc_fixtures(monkeypatch, tmp_path, mountpoint, disconnected=False)
    mount = safety.mount_for_path(str(mountpoint / "source.nii"))
    monkeypatch.setattr(
        safety,
        "_write_guard_state",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        safety._acquire_mount_guard(mount)
    assert not safety._quarantine_path(mount).exists()


def test_public_command_repeated_disconnected_failures_leave_no_owned_state(tmp_path):
    mountpoint = tmp_path / "share"
    mountinfo = tmp_path / "mountinfo"
    debug = tmp_path / "DebugData"
    runtime = tmp_path / "runtime"
    mountinfo.write_text(_mountinfo(mountpoint))
    debug.write_text(_debug_data(disconnected=True))
    env = os.environ.copy()
    env.update(
        {
            "_ARRAYVIEW_TEST_MOUNTINFO": str(mountinfo),
            "_ARRAYVIEW_TEST_CIFS_DEBUG": str(debug),
            "ARRAYVIEW_RUNTIME_DIR": str(runtime),
            "_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT": str(tmp_path / "source-safety"),
            "ARRAYVIEW_SKIP_SOURCE_STAGING": "0",
        }
    )
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    command = [
        sys.executable,
        "-m",
        "arrayview",
        str(mountpoint / "blocked.nii"),
        "--window",
        "none",
        "--port",
        str(port),
    ]
    started = time.monotonic()
    for _ in range(5):
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
        assert result.returncode == 1
        assert "disconnected" in (result.stdout + result.stderr).lower()
    assert time.monotonic() - started < 10
    assert not list((runtime / "instances").glob("*.json"))
    assert not (runtime / "startup.lock").exists()
    with socket.socket() as reusable:
        reusable.bind(("localhost", port))
    source_root = tmp_path / "source-safety"
    assert not list((source_root / "staging").glob("*"))
    assert not list((source_root / "mount-guards").glob("*"))


def test_public_command_bounds_blocked_copy_and_does_not_spawn_retries(tmp_path):
    mountpoint = tmp_path / "share"
    mountpoint.mkdir()
    source = mountpoint / "blocked.nii"
    source.write_bytes(b"copy helper sleeps before opening this")
    mountinfo = tmp_path / "mountinfo"
    debug = tmp_path / "DebugData"
    runtime = tmp_path / "runtime"
    source_root = tmp_path / "source-safety"
    mountinfo.write_text(_mountinfo(mountpoint))
    debug.write_text(_debug_data(disconnected=False))
    env = os.environ.copy()
    env.update(
        {
            "_ARRAYVIEW_TEST_MOUNTINFO": str(mountinfo),
            "_ARRAYVIEW_TEST_CIFS_DEBUG": str(debug),
            "_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT": str(source_root),
            "_ARRAYVIEW_TEST_COPY_SLEEP_SECONDS": "3600",
            "ARRAYVIEW_SOURCE_TIMEOUT_SECONDS": "0.1",
            "ARRAYVIEW_RUNTIME_DIR": str(runtime),
            "ARRAYVIEW_SKIP_SOURCE_STAGING": "0",
        }
    )
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "arrayview",
        str(source),
        "--window",
        "none",
        "--port",
        str(port),
    ]

    started = time.monotonic()
    results = [
        subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
        for _ in range(3)
    ]
    assert time.monotonic() - started < 5
    assert all(result.returncode == 1 for result in results)
    assert "timed out" in (results[0].stdout + results[0].stderr).lower()
    assert all(
        "quarantined" in (result.stdout + result.stderr).lower()
        for result in results[1:]
    )
    guards = list((source_root / "mount-guards").glob("*/state.json"))
    assert len(guards) == 1
    state = json.loads(guards[0].read_text())
    assert state["status"] == "reaped"
    from arrayview._instance_registry import process_start_identity

    assert process_start_identity(int(state["pid"])) is None
    assert not list((source_root / "staging").glob("*"))
    assert not list((runtime / "instances").glob("*.json"))


@pytest.mark.parametrize("kind", ["directory", "extensionless-dicom"])
def test_public_command_rejects_related_discovery_across_nested_network_mount(
    tmp_path, kind
):
    local_root = tmp_path / "local"
    local_root.mkdir()
    nested = local_root / "nested-network"
    source = local_root if kind == "directory" else local_root / "IM0001"
    if kind != "directory":
        source.write_bytes(b"not inspected")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "10 1 0:1 / / rw - ext4 /dev/root rw\n"
        + _mountinfo(nested)
    )
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "_ARRAYVIEW_TEST_MOUNTINFO": str(mountinfo),
            "ARRAYVIEW_RUNTIME_DIR": str(runtime),
        }
    )
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arrayview",
            str(source),
            "--window",
            "none",
            "--port",
            str(port),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 1
    assert "discovery would enter network mount" in (
        result.stdout + result.stderr
    ).lower()
    assert not list((runtime / "instances").glob("*.json"))


@pytest.mark.parametrize("network", [False, True], ids=["local", "healthy-cifs"])
def test_public_nifti_no_display_loads_and_releases_owned_state(tmp_path, network):
    nib = pytest.importorskip("nibabel")
    source_dir = tmp_path / ("share" if network else "local")
    source_dir.mkdir()
    source = source_dir / "volume.nii"
    nib.save(
        nib.Nifti1Image(np.arange(60, dtype=np.float32).reshape(3, 4, 5), np.eye(4)),
        source,
    )
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env["ARRAYVIEW_RUNTIME_DIR"] = str(runtime)
    env["_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT"] = str(tmp_path / "source-safety")
    if network:
        mountinfo = tmp_path / "mountinfo"
        debug = tmp_path / "DebugData"
        mountinfo.write_text(_mountinfo(source_dir))
        debug.write_text(_debug_data(disconnected=False))
        env["_ARRAYVIEW_TEST_MOUNTINFO"] = str(mountinfo)
        env["_ARRAYVIEW_TEST_CIFS_DEBUG"] = str(debug)
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    command = [
        sys.executable,
        "-m",
        "arrayview",
        str(source),
        "--window",
        "none",
        "--port",
        str(port),
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and list((runtime / "instances").glob("*.json")):
        time.sleep(0.05)
    assert not list((runtime / "instances").glob("*.json"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.socket() as reusable:
                reusable.bind(("localhost", port))
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("no-display daemon kept its port after releasing the session")
    assert not list((tmp_path / "source-safety" / "staging").glob("*"))


def test_public_no_display_waits_for_and_releases_related_sessions(tmp_path):
    base = tmp_path / "base.npy"
    compare = tmp_path / "compare.npy"
    overlay = tmp_path / "overlay.npy"
    np.save(base, np.zeros((4, 4), dtype=np.float32))
    np.save(compare, np.ones((4, 4), dtype=np.float32))
    np.save(overlay, np.ones((4, 4), dtype=np.uint8))
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env["ARRAYVIEW_RUNTIME_DIR"] = str(runtime)
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arrayview",
            str(base),
            str(compare),
            "--overlay",
            str(overlay),
            "--window",
            "none",
            "--port",
            str(port),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.socket() as reusable:
                reusable.bind(("localhost", port))
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("related no-display sessions kept the daemon alive")
    assert not list((runtime / "instances").glob("*.json"))


def test_session_owned_watch_thread_stops_on_release(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace
    import arrayview._lifecycle as lifecycle
    import arrayview._session as session_state
    from arrayview._watch import attach_file_watch

    source = tmp_path / "watch.npy"
    np.save(source, np.zeros((2, 2), dtype=np.float32))
    sid = "watch-session"
    session = SimpleNamespace(
        sid=sid,
        viewer_leases=1,
        related_release_sids=[],
        collection_overlay_sids=[],
        reset_caches=lambda: None,
        data=np.zeros((2, 2)),
    )
    monkeypatch.setitem(session_state.SESSIONS, sid, session)
    attach_file_watch(session, str(source), 43000)
    thread = session._source_watch_thread
    assert thread.is_alive()
    assert lifecycle.release_session(sid)
    assert not thread.is_alive()
    assert not any(
        item.ident == thread.ident
        for item in threading.enumerate()
        if item.name.startswith("arrayview-watch-")
    )


def test_final_session_release_removes_network_staging(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import arrayview._lifecycle as lifecycle
    import arrayview._session as session_state

    safe_root = tmp_path / "source-safety"
    monkeypatch.setenv("_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT", str(safe_root))
    staging = safe_root / "staging" / "owned"
    staging.mkdir(parents=True)
    (staging / "source.nii").write_bytes(b"snapshot")
    sid = "staged-session"
    session = SimpleNamespace(
        sid=sid,
        viewer_leases=1,
        related_release_sids=[],
        collection_overlay_sids=[],
        _source_staging_dirs=[str(staging)],
        reset_caches=lambda: None,
        data=np.zeros((2, 2)),
    )
    monkeypatch.setitem(session_state.SESSIONS, sid, session)

    assert lifecycle.release_session(sid)
    assert not staging.exists()
    assert sid not in session_state.SESSIONS
