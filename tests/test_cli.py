import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

import arrayview._app as appmod
import arrayview._launcher as _launcher_mod


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def _record_opened_url(opened: dict):
    def _open(url, *args, **kwargs):
        opened.setdefault("url", url)
        return url

    return _open


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _wait_for_ping(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    url = f"http://localhost:{port}/ping"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as res:
                return res.status == 200
        except Exception:
            time.sleep(0.1)
    return False


def test_cli_positional_compare_paths_register_and_open(monkeypatch, tmp_path):
    base = str(tmp_path / "base.npy")
    cmp1 = str(tmp_path / "cmp1.npy")
    cmp2 = str(tmp_path / "cmp2.npy")
    np.save(
        base, np.zeros((8, 8), dtype=np.float32)
    )  # must exist for os.path.isfile check
    requests = []
    opened = {}

    monkeypatch.setattr(sys, "argv", ["arrayview", base, cmp1, cmp2, "--browser"])
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        _record_opened_url(opened),
    )

    def fake_urlopen(req, timeout=5):
        body = json.loads((req.data or b"{}").decode())
        requests.append(body)
        fp = body.get("filepath")
        if fp == cmp1:
            return _DummyResponse({"sid": "sid_cmp1", "name": "cmp1.npy"})
        if fp == cmp2:
            return _DummyResponse({"sid": "sid_cmp2", "name": "cmp2.npy"})
        if fp == base:
            return _DummyResponse({"sid": "sid_base", "name": "base.npy"})
        return _DummyResponse({"error": f"unexpected filepath: {fp}"})

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert len(requests) == 3
    assert requests[0]["filepath"] == cmp1
    assert requests[1]["filepath"] == cmp2
    assert requests[2]["filepath"] == base
    assert opened["url"].startswith("http://localhost:8000/?sid=sid_base")
    assert "127.0.0.1" not in opened["url"]
    assert "compare_sid=sid_cmp1" in opened["url"]
    assert "compare_sids=sid_cmp1,sid_cmp2" in opened["url"]


def test_cli_vscode_terminal_keeps_spawned_backend_available_for_tab_startup(tmp_path):
    arr_path = tmp_path / "base.npy"
    np.save(arr_path, np.zeros((8, 8), dtype=np.float32))
    port = _free_port()

    env = dict(os.environ)
    env.update(
        {
            "TERM_PROGRAM": "vscode",
            "ARRAYVIEW_WINDOW_ID": "test-window",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arrayview",
            str(arr_path),
            "--window",
            "vscode",
            "--port",
            str(port),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    try:
        assert result.returncode == 0, result.stderr
        assert _wait_for_ping(port)
    finally:
        subprocess.run(
            [sys.executable, "-m", "arrayview", "--kill", "--port", str(port)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )


def test_cli_rejects_more_than_six_files(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["arrayview", "a.npy", "b.npy", "c.npy", "d.npy", "e.npy", "f.npy", "g.npy"],
    )
    with pytest.raises(SystemExit):
        appmod.arrayview()


def test_cli_accepts_six_total_files_for_compare(monkeypatch, tmp_path):
    files = [str(tmp_path / f"a{i}.npy") for i in range(6)]
    np.save(files[0], np.zeros((8, 8), dtype=np.float32))  # only base must exist
    requests = []
    opened = {}

    monkeypatch.setattr(sys, "argv", ["arrayview", *files, "--browser"])
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        _record_opened_url(opened),
    )

    def fake_urlopen(req, timeout=5):
        body = json.loads((req.data or b"{}").decode())
        requests.append(body)
        fp = body.get("filepath")
        idx = files.index(fp)
        return _DummyResponse({"sid": f"sid_{idx}", "name": f"a{idx}.npy"})

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert len(requests) == 6
    compare_sid_csv = ",".join([f"sid_{i}" for i in range(1, 6)])
    assert opened["url"].startswith("http://localhost:8000/?sid=sid_0")
    assert "127.0.0.1" not in opened["url"]
    assert "compare_sid=sid_1" in opened["url"]
    assert f"compare_sids={compare_sid_csv}" in opened["url"]


def test_cli_existing_server_overlay_and_dims_are_forwarded_to_browser(
    monkeypatch, tmp_path
):
    base = str(tmp_path / "base.npy")
    overlay = str(tmp_path / "overlay.npy")
    np.save(base, np.zeros((8, 8, 4), dtype=np.float32))
    np.save(overlay, np.ones((8, 8, 4), dtype=np.float32))
    opened = {}
    requests = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arrayview",
            base,
            "--overlay",
            overlay,
            "--dims",
            "1,2",
            "--browser",
        ],
    )
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        _record_opened_url(opened),
    )

    def fake_urlopen(req, timeout=5):
        body = json.loads((req.data or b"{}").decode())
        requests.append(body)
        fp = body.get("filepath")
        if fp == overlay:
            return _DummyResponse({"sid": "sid_overlay", "name": "overlay.npy"})
        if fp == base:
            return _DummyResponse({"sid": "sid_base", "name": "base.npy"})
        return _DummyResponse({"error": f"unexpected filepath: {fp}"})

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert [body["filepath"] for body in requests] == [overlay, base]
    assert opened["url"] == (
        "http://localhost:8000/?sid=sid_base"
        "&overlay_sid=sid_overlay"
        "&dim_x=1"
        "&dim_y=2"
    )


def test_cli_dir_dry_run_prints_collection_summary(monkeypatch, tmp_path):
    shape = (4, 5, 6)
    (tmp_path / "images").mkdir()
    (tmp_path / "gt").mkdir()
    for case in ("caseA", "caseB"):
        np.save(tmp_path / "images" / f"{case}_0000.npy", np.zeros(shape))
        np.save(tmp_path / "gt" / f"{case}.npy", np.ones(shape, dtype=np.uint8))

    stdout = io.StringIO()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arrayview",
            "--dir",
            str(tmp_path / "images" / "*_0000.npy"),
            "--overlay",
            f"gt={tmp_path / 'gt' / '*.npy'}",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    appmod.arrayview()

    text = stdout.getvalue()
    assert "--dir matched collection" in text
    assert "cases: 2" in text
    assert "overlay gt:" in text
    assert "caseA, caseB" in text


def test_cli_existing_server_native_injection_skips_browser(monkeypatch, tmp_path):
    base = str(tmp_path / "base.npy")
    np.save(base, np.zeros((8, 8), dtype=np.float32))
    opened = []

    monkeypatch.setattr(sys, "argv", ["arrayview", base, "--window", "native"])
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(_launcher_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )
    monkeypatch.setattr(
        _launcher_mod,
        "_vprint",
        lambda *args, **kwargs: None,
    )

    def fake_urlopen(req, timeout=5):
        body = json.loads((req.data or b"{}").decode())
        return _DummyResponse({"sid": "sid_base", "name": body["name"], "notified": True})

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert opened == []


def test_cli_existing_server_native_shell_connection_skips_browser(
    monkeypatch, tmp_path
):
    base = str(tmp_path / "base.npy")
    np.save(base, np.zeros((8, 8), dtype=np.float32))
    opened = []
    terminated = []

    proc = type(
        "P",
        (),
        {
            "poll": lambda self: None,
            "terminate": lambda self: terminated.append(True),
        },
    )()

    monkeypatch.setattr(sys, "argv", ["arrayview", base, "--window", "native"])
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(_launcher_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_webview_cli_tracked",
        lambda *args, **kwargs: (True, proc),
    )
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        lambda *args, **kwargs: opened.append((args, kwargs)),
    )
    monkeypatch.setattr(_launcher_mod, "_vprint", lambda *args, **kwargs: None)

    def fake_urlopen(req, timeout=5):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/ping"):
            return _DummyResponse(
                {
                    "ok": True,
                    "service": "arrayview",
                    "viewer_connections_seen": 0,
                    "viewer_sockets": 0,
                    "shell_sockets": 1,
                }
            )
        body = json.loads((req.data or b"{}").decode())
        return _DummyResponse(
            {
                "sid": "sid_base",
                "name": body["name"],
                "notified": False,
            }
        )

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert opened == []
    assert terminated == []


def test_cli_spawn_daemon_opens_viewer_url(monkeypatch, tmp_path):
    base = str(tmp_path / "base.npy")
    np.save(base, np.zeros((8, 8), dtype=np.float32))
    opened = []
    spawned = []

    class _DummyProc:
        pid = 43210

    monkeypatch.setattr(sys, "argv", ["arrayview", base, "--browser"])
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: False)
    monkeypatch.setattr(_launcher_mod, "_port_in_use", lambda _: False)
    monkeypatch.setattr(_launcher_mod, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(_launcher_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(
        _launcher_mod.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append(cmd) or _DummyProc(),
    )
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        lambda url, **kwargs: opened.append({"url": url, **kwargs}),
    )
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    assert spawned
    assert "_serve_daemon(" in spawned[0][2]
    assert len(opened) == 1
    assert opened[0]["url"].startswith("http://localhost:8000/?sid=")
    assert opened[0]["blocking"] is True
    assert opened[0]["force_vscode"] is False
    assert opened[0]["title"] == "ArrayView: base.npy"
    assert opened[0]["floating"] is False


def test_cli_vectorfield_components_dim_is_sent_to_attach(monkeypatch, tmp_path):
    base = str(tmp_path / "base.npy")
    vf = str(tmp_path / "vf.npy")
    np.save(base, np.zeros((8, 8), dtype=np.float32))
    np.save(vf, np.zeros((8, 8, 3), dtype=np.float32))
    requests = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arrayview",
            base,
            "--vectorfield",
            vf,
            "--vectorfield-components-dim",
            "2",
            "--browser",
        ],
    )
    monkeypatch.setattr(_launcher_mod, "_server_alive", lambda _: True)
    monkeypatch.setattr(
        _launcher_mod,
        "_open_browser",
        lambda url, *args, **kwargs: url,
    )

    def fake_urlopen(req, timeout=5):
        body = json.loads((req.data or b"{}").decode())
        requests.append(body)
        if req.full_url.endswith("/attach_vectorfield"):
            return _DummyResponse({"ok": True, "components_dim": 2})
        return _DummyResponse({"sid": "sid_base", "name": "base.npy"})

    monkeypatch.setattr(appmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    appmod.arrayview()

    attach_body = next(b for b in requests if b.get("filepath") == vf)
    assert attach_body["components_dim"] == 2


def test_cli_vectorfield_ambiguous_component_axis_requires_flag(monkeypatch, tmp_path):
    base = str(tmp_path / "base.npy")
    vf = str(tmp_path / "vf_amb.npy")
    np.save(base, np.zeros((6, 8, 10), dtype=np.float32))
    np.save(vf, np.zeros((3, 6, 8, 10, 3), dtype=np.float32))

    monkeypatch.setattr(
        sys,
        "argv",
        ["arrayview", base, "--vectorfield", vf, "--browser"],
    )

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    with pytest.raises(SystemExit):
        appmod.arrayview()

    assert "--vectorfield-components-dim" in out.getvalue()
