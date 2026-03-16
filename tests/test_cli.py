import io
import json
import sys

import numpy as np
import pytest

import arrayview._app as appmod
import arrayview._launcher as _launcher_mod


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


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
        lambda url, blocking=False, force_vscode=False: opened.setdefault("url", url),
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
    assert "compare_sid=sid_cmp1" in opened["url"]
    assert "compare_sids=sid_cmp1,sid_cmp2" in opened["url"]


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
        lambda url, blocking=False, force_vscode=False: opened.setdefault("url", url),
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
    assert "compare_sid=sid_1" in opened["url"]
    assert f"compare_sids={compare_sid_csv}" in opened["url"]
