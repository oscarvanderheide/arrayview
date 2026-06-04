from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_local_vscode_spawned_daemon_uses_transient_backend(monkeypatch):
    import arrayview._launcher as launcher

    spawned = []

    monkeypatch.setattr(launcher, "_configure_vscode_port_preview", lambda port: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
    monkeypatch.setattr(launcher, "_open_cli_spawned_view", lambda **kwargs: None)
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid"})())

    launcher._handle_cli_spawned_daemon(
        port=8000,
        base_file="/tmp/base.npy",
        name="base.npy",
        compare_files=[],
        overlay_files=[],
        dims_override=None,
        use_webview=False,
        watch=False,
        window_mode="vscode",
        floating=False,
        is_remote=False,
        vectorfield=None,
        vfield_components_dim=None,
        rgb=False,
        demo_name=None,
        demo_cleanup=False,
    )

    assert spawned
    assert "persist=False" in spawned[0][0][2]


def test_remote_vscode_spawned_daemon_keeps_backend_persistent(monkeypatch):
    import arrayview._launcher as launcher

    spawned = []

    monkeypatch.setattr(launcher, "_configure_vscode_port_preview", lambda port: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
    monkeypatch.setattr(launcher, "_open_cli_spawned_view", lambda **kwargs: None)
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid"})())

    launcher._handle_cli_spawned_daemon(
        port=8000,
        base_file="/tmp/base.npy",
        name="base.npy",
        compare_files=[],
        overlay_files=[],
        dims_override=None,
        use_webview=False,
        watch=False,
        window_mode="vscode",
        floating=False,
        is_remote=True,
        vectorfield=None,
        vfield_components_dim=None,
        rgb=False,
        demo_name=None,
        demo_cleanup=False,
    )

    assert spawned
    assert "persist=True" in spawned[0][0][2]


def test_viewer_sid_tracking_clears_on_websocket_disconnect(tmp_path):
    import arrayview._session as session_mod
    from arrayview._app import app

    session_mod.VIEWER_SOCKETS = 0
    session_mod.VIEWER_SIDS.clear()
    session_mod.VIEWER_SID_COUNTS.clear()

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "viewer_sid.npy", arr)

    with TestClient(app) as client:
        sid = client.post(
            "/load", json={"filepath": str(tmp_path / "viewer_sid.npy")}
        ).json()["sid"]

        with client.websocket_connect(f"/ws/{sid}") as ws:
            assert ws.receive_json()["type"] == "metadata"
            assert session_mod.VIEWER_SOCKETS == 1
            assert sid in session_mod.VIEWER_SIDS
            assert session_mod.VIEWER_SID_COUNTS[sid] == 1

        assert session_mod.VIEWER_SOCKETS == 0
        assert sid not in session_mod.VIEWER_SIDS
        assert sid not in session_mod.VIEWER_SID_COUNTS


def test_release_route_drops_session_and_is_idempotent(tmp_path):
    from arrayview._app import app

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "release.npy", arr)

    with TestClient(app) as client:
        sid = client.post("/load", json={"filepath": str(tmp_path / "release.npy")}).json()["sid"]
        assert client.get(f"/metadata/{sid}").status_code == 200

        released = client.post(f"/release/{sid}")
        assert released.status_code == 200
        assert released.json() == {"sid": sid, "released": True}
        assert client.get(f"/metadata/{sid}").status_code == 404

        released_again = client.post(f"/release/{sid}")
        assert released_again.status_code == 200
        assert released_again.json() == {"sid": sid, "released": False}


def test_shell_close_uses_release_route_semantics(tmp_path):
    from arrayview._app import app

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "shell-release.npy", arr)

    with TestClient(app) as client:
        sid = client.post(
            "/load", json={"filepath": str(tmp_path / "shell-release.npy")}
        ).json()["sid"]
        assert client.get(f"/metadata/{sid}").status_code == 200

        with client.websocket_connect("/ws/shell") as ws:
            ws.send_json({"action": "close", "sid": sid})

        assert client.get(f"/metadata/{sid}").status_code == 404


def test_vscode_wrapper_backend_check_uses_extension_host_ping():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "pingUrl = `${parsed.origin}/ping`" in source
    assert "await httpOk(pingUrl)" in source
    assert "viewerReady = true" in source
    assert "postMessage({ type: 'backend-error', url })" in source
    assert "fetch(pingUrl" not in source


def test_vscode_url_panel_dispose_releases_primary_sid():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "function releaseUrlSession(url)" in source
    assert "parsed.searchParams.get('sid')" in source
    assert "/release/${encodeURIComponent(sid)}" in source
    assert "releaseUrlSession(url)" in source
