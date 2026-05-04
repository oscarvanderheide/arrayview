"""Layer 1: HTTP API tests (no browser required, runs in seconds)."""

import base64
import io
import inspect
import os
import subprocess
import sys
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_ping(self, client):
        r = client.get("/ping")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["service"] == "arrayview"

    def test_root_without_sid_returns_html(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_with_sid_returns_viewer(self, client, sid_2d):
        r = client.get(f"/?sid={sid_2d}")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_root_with_sid_includes_proxy_base_support(self, client, sid_2d):
        r = client.get(f"/?sid={sid_2d}")
        assert r.status_code == 200
        assert "resolveServerPath(path)" in r.text
        assert "window.location.pathname.match(/^(.*\\/proxy\\/\\d+)(?:\\/|$)/)" in r.text
        assert '<script src="gsap.min.js"></script>' in r.text

    def test_startup_overlay_has_no_artificial_dwell(self, client, sid_2d):
        r = client.get(f"/?sid={sid_2d}")
        assert r.status_code == 200
        assert "const _MIN_SPINNER_MS = 0" in r.text

    def test_shell_returns_html(self, client):
        r = client.get("/shell")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "document.createElement('iframe')" in r.text
        assert '<meta name="color-scheme" content="dark">' in r.text
        assert "function hideShellLoadingOverlay()" in r.text
        assert "phase === 'script-loaded' || phase === 'frame-rendered'" in r.text
        assert "skip the overlay entirely" not in r.text

    def test_launcher_cold_start_loading_infrastructure(self):
        import arrayview._launcher as launcher

        html = launcher._LOADING_HTML

        assert "#0c0c0c" in html  # dark background matches viewer theme
        assert "window.location.replace" in html  # JS navigates when server ready
        assert callable(launcher._run_loading_server)
        assert callable(launcher._with_loading)

    def test_package_import_keeps_startup_modules_lazy(self):
        code = (
            "import sys, arrayview; "
            "print('arrayview._session' in sys.modules, "
            "'arrayview._vscode' in sys.modules, "
            "'numpy' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "False False False"

    def test_metadata_default_dims_match_viewer_startup_for_4d_data(self, client, tmp_path):
        arr = (np.random.randn(22, 24, 21, 5) + 1j * np.random.randn(22, 24, 21, 5)).astype(np.complex64)
        path = tmp_path / "startup_dims_complex.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        body = client.get(f"/metadata/{sid}").json()

        assert body["default_dims"] == [0, 1]

    def test_sessions_lists_registered_sid(self, client, sid_2d):
        r = client.get("/sessions")
        assert r.status_code == 200
        sids = [s["sid"] for s in r.json()]
        assert sid_2d in sids


# ---------------------------------------------------------------------------
# /load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_2d_returns_sid_and_name(self, client, arr_2d, tmp_path):
        np.save(tmp_path / "a.npy", arr_2d)
        r = client.post(
            "/load", json={"filepath": str(tmp_path / "a.npy"), "name": "myarray"}
        )
        assert r.status_code == 200
        body = r.json()
        assert "sid" in body
        assert body["name"] == "myarray"

    def test_load_missing_file_returns_error(self, client):
        r = client.post("/load", json={"filepath": "/nonexistent/path/arr.npy"})
        assert r.status_code == 200  # endpoint returns 200 with error key
        assert "error" in r.json()

    def test_load_name_defaults_to_filename(self, client, arr_2d, tmp_path):
        np.save(tmp_path / "coolarray.npy", arr_2d)
        r = client.post("/load", json={"filepath": str(tmp_path / "coolarray.npy")})
        assert r.json()["name"] == "coolarray.npy"

    def test_notify_existing_session_returns_open_state(self, client, sid_2d):
        r = client.post(
            f"/notify/{sid_2d}",
            json={"name": "already-loaded", "url": "/?sid=abc", "wait": False},
        )

        assert r.status_code == 200
        body = r.json()
        assert body == {
            "sid": sid_2d,
            "name": "already-loaded",
            "notified": False,
        }

    def test_notify_missing_session_is_404(self, client):
        r = client.post("/notify/missing", json={})

        assert r.status_code == 404

    def test_wait_for_session_ready_uses_pending_event(self):
        import asyncio

        import arrayview._session as session_mod

        sid = "pending-event-test"
        marker = object()
        event = threading.Event()
        session_mod.PENDING_SESSIONS.add(sid)
        session_mod.PENDING_SESSION_EVENTS[sid] = event

        def publish():
            time.sleep(0.01)
            session_mod.SESSIONS[sid] = marker
            event.set()

        try:
            threading.Thread(target=publish, daemon=True).start()
            t0 = time.perf_counter()
            result = asyncio.run(session_mod.wait_for_session_ready(sid, timeout=1.0))
            elapsed = time.perf_counter() - t0
        finally:
            session_mod.SESSIONS.pop(sid, None)
            session_mod.PENDING_SESSIONS.discard(sid)
            session_mod.PENDING_SESSION_EVENTS.pop(sid, None)

        assert result is marker
        assert elapsed < 0.08


class TestFsList:
    def test_fs_list_filters_supported_entries_and_overlay_shapes(
        self, client, sid_2d, tmp_path, monkeypatch
    ):
        import arrayview._routes_loading as routes_loading

        monkeypatch.setattr(routes_loading.os.path, "expanduser", lambda _p: str(tmp_path))

        good = tmp_path / "good.npy"
        bad = tmp_path / "bad.npy"
        note = tmp_path / "note.txt"
        child = tmp_path / "child"
        child.mkdir()

        np.save(good, np.zeros((100, 80), dtype=np.float32))
        np.save(bad, np.zeros((100, 81), dtype=np.float32))
        note.write_text("ignore me")

        r = client.get(
            "/fs/list",
            params={"path": str(tmp_path), "base_sid": sid_2d, "mode": "overlay"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["cwd"] == str(tmp_path)
        names = [entry["name"] for entry in body["entries"]]
        assert "child" in names
        assert "good.npy" in names
        assert "bad.npy" not in names
        assert "note.txt" not in names


class TestLoadBytes:
    def test_load_bytes_creates_session_and_uses_localhost_url(self, client, monkeypatch):
        import arrayview._session as session_mod
        import arrayview._vscode as vscode

        opened = []
        monkeypatch.setattr(vscode, "_open_via_signal_file", opened.append)
        monkeypatch.setattr(session_mod, "SERVER_PORT", 8123)

        buf = io.BytesIO()
        np.save(buf, np.arange(12, dtype=np.float32).reshape(3, 4))
        payload = base64.b64encode(buf.getvalue()).decode("ascii")

        r = client.post("/load_bytes", json={"data_b64": payload, "name": "remote.npy"})
        assert r.status_code == 200
        body = r.json()
        assert body["url"] == f"http://localhost:8123/?sid={body['sid']}"
        assert opened == [body["url"]]


@pytest.fixture
def fake_segmentation(monkeypatch):
    import arrayview._segmentation as seg
    import arrayview._routes_segmentation as seg_routes

    state = {
        "connected": False,
        "uploaded": None,
        "point_calls": [],
        "scribble_calls": [],
        "reset_calls": 0,
        "disconnect_calls": 0,
    }

    def reset_server_state():
        if seg_routes._seg_overlay_sid is not None:
            seg_routes.SESSIONS.pop(seg_routes._seg_overlay_sid, None)
        seg_routes._seg_overlay_sid = None
        seg_routes._seg_label_mask = None
        seg_routes._seg_current_label = 0
        seg_routes._seg_vol_axes = None
        seg_routes._seg_fixed_indices = None
        seg_routes._seg_full_shape = None
        seg_routes._seg_label_names.clear()

    reset_server_state()

    def is_connected():
        return state["connected"]

    def try_connect(host=seg.DEFAULT_HOST, port=seg.DEFAULT_PORT, *, url=None):
        state["connected"] = True
        state["connect_args"] = {"host": host, "port": port, "url": url}
        return True

    def try_launch(host=seg.DEFAULT_HOST, port=seg.DEFAULT_PORT):
        state["connected"] = True
        state["launch_args"] = {"host": host, "port": port}
        return None

    def upload_volume(data):
        state["uploaded"] = np.array(data, copy=True)

    def add_point(coord_zyx, positive=True):
        state["point_calls"].append((tuple(coord_zyx), positive))
        mask = np.zeros(state["uploaded"].shape, dtype=np.uint8)
        mask[tuple(coord_zyx)] = 1
        return mask

    def add_scribble(mask_3d, positive=True):
        state["scribble_calls"].append((np.array(mask_3d, copy=True), positive))
        return np.array(mask_3d, copy=True)

    def reset_interactions():
        state["reset_calls"] += 1

    def disconnect():
        state["connected"] = False
        state["disconnect_calls"] += 1

    monkeypatch.setattr(seg, "is_connected", is_connected)
    monkeypatch.setattr(seg, "try_connect", try_connect)
    monkeypatch.setattr(seg, "try_launch", try_launch)
    monkeypatch.setattr(seg, "upload_volume", upload_volume)
    monkeypatch.setattr(seg, "add_point", add_point)
    monkeypatch.setattr(seg, "add_scribble", add_scribble)
    monkeypatch.setattr(seg, "reset_interactions", reset_interactions)
    monkeypatch.setattr(seg, "disconnect", disconnect)

    yield state

    reset_server_state()


class TestSegmentation:
    def test_activate_uploads_selected_4d_subvolume(
        self, client, sid_4d, arr_4d, fake_segmentation
    ):
        r = client.post(
            f"/seg/activate/{sid_4d}",
            params={
                "dim_x": 1,
                "dim_y": 2,
                "scroll_dim": 3,
                "indices": "2,0,0,0",
            },
        )

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["message"] == "connected"
        assert body["ndim"] == 4
        np.testing.assert_array_equal(fake_segmentation["uploaded"], arr_4d[2])

    def test_scribble_rasterizes_current_slice_and_updates_overlay(
        self, client, sid_3d, arr_3d, fake_segmentation
    ):
        activated = client.post(f"/seg/activate/{sid_3d}")
        assert activated.status_code == 200
        assert activated.json()["status"] == "ok"

        r = client.post(
            f"/seg/scribble/{sid_3d}",
            json={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "3,10,12",
                "points": [[5, 6], [9, 6], [9, 10]],
                "positive": False,
            },
        )

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

        import arrayview._routes_segmentation as seg_routes

        assert len(fake_segmentation["scribble_calls"]) == 1
        rasterized, positive = fake_segmentation["scribble_calls"][0]
        assert positive is False
        assert rasterized.shape == arr_3d.shape
        assert np.count_nonzero(rasterized) == np.count_nonzero(rasterized[3])
        assert np.count_nonzero(rasterized[3]) > 0
        overlay_sid = body["overlay_sid"]
        assert overlay_sid in seg_routes.SESSIONS
        np.testing.assert_array_equal(seg_routes.SESSIONS[overlay_sid].data, rasterized)

    def test_click_accept_labels_export_delete_and_disconnect_roundtrip(
        self, client, sid_3d, arr_3d, fake_segmentation
    ):
        activated = client.post(f"/seg/activate/{sid_3d}")
        assert activated.status_code == 200
        assert activated.json()["status"] == "ok"

        clicked = client.post(
            f"/seg/click/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "4,7,9",
                "px": 11,
                "py": 12,
                "positive": True,
            },
        )
        assert clicked.status_code == 200
        assert clicked.json()["status"] == "ok"
        assert fake_segmentation["point_calls"] == [((4, 12, 11), True)]

        accepted = client.post(f"/seg/accept/{sid_3d}")
        assert accepted.status_code == 200
        accepted_body = accepted.json()
        assert accepted_body["status"] == "ok"
        assert accepted_body["labels"] == [
            {
                "label": 1,
                "name": "segment 1",
                "color": "#ff5050",
                "voxels": 1,
            }
        ]
        assert fake_segmentation["reset_calls"] == 1

        renamed = client.post(
            f"/seg/rename/{sid_3d}", params={"label": 1, "name": "tumor"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["status"] == "ok"

        labels = client.get(f"/seg/labels/{sid_3d}")
        assert labels.status_code == 200
        labels_body = labels.json()
        assert labels_body["labels"] == [
            {
                "label": 1,
                "name": "tumor",
                "color": "#ff5050",
                "voxels": 1,
            }
        ]

        exported = client.get(f"/seg/export/{sid_3d}")
        assert exported.status_code == 200
        assert exported.headers["content-disposition"] == (
            "attachment; filename=segmentation.npy"
        )
        exported_mask = np.load(io.BytesIO(exported.content))
        assert exported_mask.shape == arr_3d.shape
        assert int(exported_mask[4, 12, 11]) == 1
        assert int(exported_mask.sum()) == 1

        deleted = client.post(f"/seg/delete_label/{sid_3d}", params={"label": 1})
        assert deleted.status_code == 200
        deleted_body = deleted.json()
        assert deleted_body["status"] == "ok"
        assert deleted_body["labels"] == []

        disconnected = client.post("/seg/disconnect")
        assert disconnected.status_code == 200
        assert disconnected.json() == {"status": "ok"}
        assert fake_segmentation["disconnect_calls"] == 1


# ---------------------------------------------------------------------------
# /metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_2d_shape_and_not_complex(self, client, sid_2d):
        r = client.get(f"/metadata/{sid_2d}")
        assert r.status_code == 200
        body = r.json()
        assert body["shape"] == [100, 80]
        assert body["is_complex"] is False

    def test_3d_shape(self, client, sid_3d):
        r = client.get(f"/metadata/{sid_3d}")
        assert r.status_code == 200
        assert r.json()["shape"] == [20, 64, 64]

    def test_4d_memmap_metadata_prefers_trailing_dims_for_strided_startup(
        self, client, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("AV_NPY_EAGER_BYTES", "1")
        arr = np.zeros((8, 6, 6, 6), dtype=np.float32)
        path = tmp_path / "memmap_4d.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        body = client.get(f"/metadata/{sid}").json()

        assert body["shape"] == [8, 6, 6, 6]
        assert body["default_dims"] == [2, 3]

    def test_4d_memmap_metadata_keeps_legacy_dims_when_trailing_plane_is_small(
        self, client, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("AV_NPY_EAGER_BYTES", "1")
        arr = np.zeros((8, 8, 3, 3), dtype=np.float32)
        path = tmp_path / "memmap_small_trailing.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        body = client.get(f"/metadata/{sid}").json()

        assert body["shape"] == [8, 8, 3, 3]
        assert body["default_dims"] == [0, 1]

    def test_unknown_sid_is_404(self, client):
        r = client.get("/metadata/doesnotexist000")
        assert r.status_code == 404

    def test_timed_vectorfield_metadata_and_slice(self, client, tmp_path):
        arr = np.zeros((6, 8, 10), dtype=np.float32)
        vf = np.zeros((4, 6, 8, 10, 3), dtype=np.float32)
        vf[2, :, :, :, 1] = 0.25
        vf[2, :, :, :, 2] = 0.5

        arr_path = tmp_path / "base.npy"
        vf_path = tmp_path / "vf_time.npy"
        np.save(arr_path, arr)
        np.save(vf_path, vf)

        sid = client.post(
            "/load", json={"filepath": str(arr_path), "name": "base"}
        ).json()["sid"]
        attach = client.post(
            "/attach_vectorfield", json={"sid": sid, "filepath": str(vf_path)}
        )
        assert attach.status_code == 200
        assert attach.json()["ok"] is True

        meta = client.get(f"/metadata/{sid}")
        assert meta.status_code == 200
        body = meta.json()
        assert body["has_vectorfield"] is True
        assert body["vfield_n_times"] == 4

        r = client.get(
            f"/vectorfield/{sid}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "3,4,5",
                "t_index": 2,
                "density_offset": 0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["stride"] >= 1
        assert len(data["arrows"]) > 0
        assert any(abs(a[2]) > 0 or abs(a[3]) > 0 for a in data["arrows"])

    def test_vectorfield_auto_detects_unique_non_trailing_component_axis(
        self, client, tmp_path
    ):
        arr = np.zeros((6, 8, 10), dtype=np.float32)
        vf = np.zeros((6, 3, 8, 10), dtype=np.float32)
        vf[:, 1, :, :] = 0.25
        vf[:, 2, :, :] = 0.5

        arr_path = tmp_path / "base_auto.npy"
        vf_path = tmp_path / "vf_auto.npy"
        np.save(arr_path, arr)
        np.save(vf_path, vf)

        sid = client.post(
            "/load", json={"filepath": str(arr_path), "name": "base"}
        ).json()["sid"]
        attach = client.post(
            "/attach_vectorfield", json={"sid": sid, "filepath": str(vf_path)}
        )
        assert attach.status_code == 200
        assert attach.json()["ok"] is True
        assert attach.json()["components_dim"] == 1

        r = client.get(
            f"/vectorfield/{sid}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "3,4,5",
                "t_index": 0,
                "density_offset": 0,
            },
        )
        assert r.status_code == 200
        assert len(r.json()["arrows"]) > 0

    def test_vectorfield_attach_errors_when_multiple_size3_axes_exist(
        self, client, tmp_path
    ):
        arr = np.zeros((6, 8, 10), dtype=np.float32)
        vf = np.zeros((3, 6, 8, 10, 3), dtype=np.float32)

        arr_path = tmp_path / "base_amb.npy"
        vf_path = tmp_path / "vf_amb.npy"
        np.save(arr_path, arr)
        np.save(vf_path, vf)

        sid = client.post(
            "/load", json={"filepath": str(arr_path), "name": "base"}
        ).json()["sid"]
        attach = client.post(
            "/attach_vectorfield", json={"sid": sid, "filepath": str(vf_path)}
        )
        assert attach.status_code == 200
        assert "error" in attach.json()
        assert "--vectorfield-components-dim" in attach.json()["error"]

        attach2 = client.post(
            "/attach_vectorfield",
            json={"sid": sid, "filepath": str(vf_path), "components_dim": 4},
        )
        assert attach2.status_code == 200
        assert attach2.json()["ok"] is True
        assert attach2.json()["components_dim"] == 4

    def test_oblique_vectorfield_returns_projected_arrows(self, client, tmp_path):
        arr = np.zeros((6, 8, 10), dtype=np.float32)
        vf = np.zeros((4, 6, 8, 10, 3), dtype=np.float32)
        vf[2, :, :, :, 1] = 0.25
        vf[2, :, :, :, 2] = 0.5

        arr_path = tmp_path / "base_oblique.npy"
        vf_path = tmp_path / "vf_oblique.npy"
        np.save(arr_path, arr)
        np.save(vf_path, vf)

        sid = client.post(
            "/load", json={"filepath": str(arr_path), "name": "base"}
        ).json()["sid"]
        attach = client.post(
            "/attach_vectorfield", json={"sid": sid, "filepath": str(vf_path)}
        )
        assert attach.status_code == 200
        assert attach.json()["ok"] is True

        r = client.get(
            f"/oblique_vectorfield/{sid}",
            params={
                "center": "3,4,5",
                "basis_h": "0,0,1",
                "basis_v": "0,1,0",
                "mv_dims": "0,1,2",
                "size_w": 64,
                "size_h": 64,
                "t_index": 2,
                "density_offset": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["stride"] >= 1
        assert len(body["arrows"]) > 0
        assert any(abs(a[2]) > 0 or abs(a[3]) > 0 for a in body["arrows"])


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_2d_fields(self, client, sid_2d):
        r = client.get(f"/info/{sid_2d}")
        assert r.status_code == 200
        body = r.json()
        assert body["shape"] == [100, 80]
        assert body["ndim"] == 2
        assert body["is_complex"] is False
        assert "dtype" in body
        assert "size_mb" in body

    def test_4d_ndim(self, client, sid_4d):
        r = client.get(f"/info/{sid_4d}")
        assert r.json()["ndim"] == 4

    def test_unknown_sid_is_404(self, client):
        r = client.get("/info/doesnotexist000")
        assert r.status_code == 404

    def test_recommended_colormap_and_reason_present(self, client, sid_2d):
        """Enhanced info overlay: recommended_colormap and reason are returned."""
        body = client.get(f"/info/{sid_2d}").json()
        assert "recommended_colormap" in body
        assert "recommended_colormap_reason" in body
        assert isinstance(body["recommended_colormap"], str)
        assert isinstance(body["recommended_colormap_reason"], str)


class TestThumbnail:
    def test_thumbnail_returns_jpeg(self, client, sid_2d):
        r = client.get(f"/thumbnail/{sid_2d}", params={"w": 240, "h": 180})
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content))
        assert img.size[0] <= 240
        assert img.size[1] <= 180

    def test_thumbnail_preserves_aspect_ratio_within_box(self, client, tmp_path):
        arr = np.zeros((20, 10), dtype=np.float32)
        path = tmp_path / "thumb_aspect.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        r = client.get(f"/thumbnail/{sid}", params={"w": 300, "h": 300})
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert max(img.size) == 300
        assert min(img.size) == 150

    def test_thumbnail_supports_rgb_sessions(self, client, tmp_path):
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.linspace(0, 255, 32, dtype=np.uint8)
        rgb[..., 1] = 64
        rgb[..., 2] = 200
        path = tmp_path / "rgb_thumb.npy"
        np.save(path, rgb)

        sid = client.post("/load", json={"filepath": str(path), "rgb": True}).json()["sid"]
        r = client.get(f"/thumbnail/{sid}", params={"w": 160, "h": 120})
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        px = np.array(img)[img.size[1] // 2, img.size[0] // 2]
        assert int(px[2]) > int(px[1])

    def test_signed_data_reason_mentions_rdbu(self, client, tmp_path):
        """Signed float data (vmin < 0) → colormap reason mentions RdBu_r."""
        arr = np.random.randn(20, 20).astype(np.float32)
        np.save(tmp_path / "signed.npy", arr)
        r = client.post("/load", json={"filepath": str(tmp_path / "signed.npy")})
        sid = r.json()["sid"]
        body = client.get(f"/info/{sid}").json()
        assert "RdBu_r" in body["recommended_colormap_reason"]

    def test_bool_data_reason_mentions_bool(self, client, tmp_path):
        """Bool array → colormap reason mentions bool dtype."""
        arr = np.array([[True, False], [False, True]])
        np.save(tmp_path / "bool.npy", arr)
        r = client.post("/load", json={"filepath": str(tmp_path / "bool.npy")})
        sid = r.json()["sid"]
        body = client.get(f"/info/{sid}").json()
        assert "bool" in body["recommended_colormap_reason"]

    def test_positive_data_reason_mentions_gray(self, client, tmp_path):
        """Positive float data → colormap reason mentions gray."""
        arr = np.abs(np.random.randn(20, 20).astype(np.float32)) + 1.0
        np.save(tmp_path / "pos.npy", arr)
        r = client.post("/load", json={"filepath": str(tmp_path / "pos.npy")})
        sid = r.json()["sid"]
        body = client.get(f"/info/{sid}").json()
        assert "gray" in body["recommended_colormap_reason"]


class TestExportRoutes:
    def test_export_array_returns_npy_attachment(self, client, sid_2d, arr_2d):
        r = client.get(f"/export_array/{sid_2d}")
        assert r.status_code == 200
        assert r.headers["content-disposition"] == 'attachment; filename="arr2d.npy"'
        exported = np.load(io.BytesIO(r.content))
        np.testing.assert_array_equal(exported, arr_2d)

    def test_save_file_writes_to_downloads_fallback(self, client, tmp_path, monkeypatch):
        import pathlib

        downloads = tmp_path / "Downloads"
        downloads.mkdir()
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        payload = base64.b64encode(b"hello export").decode("ascii")
        r = client.post(
            "/save_file",
            json={
                "filename": "note.txt",
                "data": f"data:text/plain;base64,{payload}",
            },
        )

        assert r.status_code == 200
        saved_path = downloads / "note.txt"
        assert r.json()["path"] == str(saved_path)
        assert saved_path.read_bytes() == b"hello export"

    def test_exploded_returns_requested_slice_previews(self, client, sid_3d):
        r = client.post(
            f"/exploded/{sid_3d}",
            json={
                "dim_x": 2,
                "dim_y": 1,
                "scroll_dim": 0,
                "indices": [0, 5, 19],
                "width": 64,
                "colormap": "gray",
                "dr": 0,
                "complex_mode": 0,
                "log_scale": False,
            },
        )

        assert r.status_code == 200
        body = r.json()
        assert [entry["index"] for entry in body["slices"]] == [0, 5, 19]
        assert len(body["slices"]) == 3
        for entry in body["slices"]:
            assert entry["image"].startswith("data:image/jpeg;base64,")
            encoded = entry["image"].split(",", 1)[1]
            assert base64.b64decode(encoded).startswith(b"\xff\xd8")


class TestPreloadRoutes:
    def test_preload_starts_and_status_reports_progress(self, client, sid_3d, monkeypatch):
        import arrayview._routes_preload as preload_routes
        from arrayview._app import SESSIONS

        def fake_run_preload(
            session,
            gen,
            dim_x,
            dim_y,
            idx_list,
            colormap,
            dr,
            slice_dim,
            dim_z,
            complex_mode,
            log_scale,
        ):
            with session.preload_lock:
                session.preload_done = 3
                session.preload_total = 7
                session.preload_skipped = False

        class ImmediateThread:
            def __init__(self, target, args, daemon):
                self._target = target
                self._args = args
                self.daemon = daemon

            def start(self):
                self._target(*self._args)

        monkeypatch.setattr(preload_routes, "_run_preload", fake_run_preload)
        monkeypatch.setattr(preload_routes.threading, "Thread", ImmediateThread)

        session = SESSIONS[sid_3d]
        session.preload_done = 0
        session.preload_total = 0
        session.preload_skipped = False

        started = client.post(
            f"/preload/{sid_3d}",
            json={
                "dim_x": 2,
                "dim_y": 1,
                "indices": [0, 0, 0],
                "slice_dim": 0,
                "dr": 0,
            },
        )
        assert started.status_code == 200
        assert started.json() == {"status": "started"}
        assert session.preload_gen >= 1

        status = client.get(f"/preload_status/{sid_3d}")
        assert status.status_code == 200
        assert status.json() == {"done": 3, "total": 7, "skipped": False}


# ---------------------------------------------------------------------------
# /slice
# ---------------------------------------------------------------------------


class TestSlice:
    def test_2d_returns_jpeg_with_correct_dimensions(self, client, sid_2d):
        r = client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content))
        # PIL reports (width, height); array is 100 rows × 80 cols
        assert img.size == (80, 100)

    def test_gray_vs_viridis_differ(self, client, sid_2d):
        base = {"dim_x": 1, "dim_y": 0, "indices": "0,0", "dr": 0, "slice_dim": 0}
        r_gray = client.get(f"/slice/{sid_2d}", params={**base, "colormap": "gray"})
        r_viridis = client.get(
            f"/slice/{sid_2d}", params={**base, "colormap": "viridis"}
        )
        assert r_gray.content != r_viridis.content

    def test_constant_array_nearly_uniform_colors(self, client, tmp_path, server_url):
        """A flat (constant) array should render as a single color."""
        arr = np.ones((40, 40), dtype=np.float32)
        np.save(tmp_path / "flat.npy", arr)
        with httpx.Client(base_url=server_url, timeout=15) as c:
            sid = c.post("/load", json={"filepath": str(tmp_path / "flat.npy")}).json()[
                "sid"
            ]
            r = c.get(
                f"/slice/{sid}",
                params={
                    "dim_x": 1,
                    "dim_y": 0,
                    "indices": "0,0",
                    "colormap": "gray",
                    "dr": 0,
                    "slice_dim": 0,
                },
            )
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        pixels = np.array(img).reshape(-1, 3)
        # Allow small JPEG compression variance
        assert int((pixels.max(axis=0) - pixels.min(axis=0)).max()) < 5

    def test_3d_slice_returns_jpeg(self, client, sid_3d):
        r = client.get(
            f"/slice/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content))
        assert img.size == (64, 64)

    def test_unknown_sid_is_404(self, client):
        r = client.get(
            "/slice/doesnotexist000",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        assert r.status_code == 404

    def test_overlay_sid_changes_http_slice(self, client, tmp_path):
        base = np.zeros((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:28, 12:28] = 1
        np.save(tmp_path / "base.npy", base)
        np.save(tmp_path / "mask.npy", mask)

        sid = client.post(
            "/load", json={"filepath": str(tmp_path / "base.npy")}
        ).json()["sid"]
        overlay_sid = client.post(
            "/load", json={"filepath": str(tmp_path / "mask.npy"), "name": "overlay"}
        ).json()["sid"]

        params = {
            "dim_x": 1,
            "dim_y": 0,
            "indices": "0,0",
            "colormap": "gray",
            "dr": 0,
            "slice_dim": 0,
        }
        plain = client.get(f"/slice/{sid}", params=params)
        over = client.get(
            f"/slice/{sid}", params={**params, "overlay_sid": overlay_sid}
        )
        assert plain.status_code == 200
        assert over.status_code == 200
        assert plain.content != over.content

        img = np.array(Image.open(io.BytesIO(over.content)).convert("RGB"))
        c = img[20, 20]
        assert c[0] > c[1] + 20
        assert c[0] > c[2] + 20


class TestProjection:
    """Tests for statistical projection rendering (p key feature)."""

    def test_max_projection_returns_jpeg(self, client, sid_3d):
        r = client.get(
            f"/slice/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "colormap": "gray",
                "dr": 0,
                "projection_mode": 1,  # MAX
                "projection_dim": 0,  # project along dim 0
            },
        )
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]

    def test_projection_modes_produce_different_images(self, client, sid_3d):
        base = {
            "dim_x": 2,
            "dim_y": 1,
            "indices": "0,0,0",
            "colormap": "gray",
            "dr": 0,
            "projection_dim": 0,
        }
        images = {}
        for mode in [1, 2, 3, 4, 5]:  # MAX, MIN, MEAN, STD, SOS
            r = client.get(
                f"/slice/{sid_3d}",
                params={**base, "projection_mode": mode},
            )
            assert r.status_code == 200
            images[mode] = r.content
        # MAX and MIN should differ
        assert images[1] != images[2]
        # MEAN and STD should differ
        assert images[3] != images[4]

    def test_projection_vs_normal_slice_differ(self, client, sid_3d):
        base = {
            "dim_x": 2,
            "dim_y": 1,
            "indices": "0,0,0",
            "colormap": "gray",
            "dr": 0,
        }
        normal = client.get(f"/slice/{sid_3d}", params=base)
        proj = client.get(
            f"/slice/{sid_3d}",
            params={**base, "projection_mode": 1, "projection_dim": 0},
        )
        assert normal.content != proj.content


class TestOverlayWebSocket:
    def test_ws_pushes_metadata_before_binary_frames(self, tmp_path):
        from arrayview._app import app

        arr = np.arange(36, dtype=np.float32).reshape(6, 6)
        np.save(tmp_path / "base_meta_ws.npy", arr)

        with TestClient(app) as c:
            sid = c.post(
                "/load", json={"filepath": str(tmp_path / "base_meta_ws.npy")}
            ).json()["sid"]

            with c.websocket_connect(f"/ws/{sid}") as ws:
                meta = ws.receive_json()

                assert meta["type"] == "metadata"
                assert meta["shape"] == [6, 6]
                assert meta["is_complex"] is False
                assert meta["is_rgb"] is False
                assert meta["has_source_file"] is True

                ws.send_json(
                    {
                        "seq": 7,
                        "dim_x": 1,
                        "dim_y": 0,
                        "dim_z": -1,
                        "indices": [0, 0],
                        "colormap": "gray",
                        "dr": 0,
                        "complex_mode": 0,
                        "log_scale": False,
                        "slice_dim": 0,
                        "direction": 1,
                    }
                )
                payload = ws.receive_bytes()

        seq, width, height = np.frombuffer(payload[:12], dtype=np.uint32)
        assert (int(seq), int(width), int(height)) == (7, 6, 6)

    def test_shell_close_drops_session(self, tmp_path):
        from arrayview._app import app

        arr = np.ones((4, 4), dtype=np.float32)
        np.save(tmp_path / "shell_close.npy", arr)

        with TestClient(app) as c:
            sid = c.post(
                "/load", json={"filepath": str(tmp_path / "shell_close.npy")}
            ).json()["sid"]

            assert c.get(f"/metadata/{sid}").status_code == 200

            with c.websocket_connect("/ws/shell") as ws:
                ws.send_json({"action": "close", "sid": sid})

            assert c.get(f"/metadata/{sid}").status_code == 404

    def test_overlay_visible_over_transparent_base(self, tmp_path):
        from arrayview._app import app

        base = np.zeros((6, 6), dtype=np.float32)
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:5, 1:5] = 1
        np.save(tmp_path / "base_ws.npy", base)
        np.save(tmp_path / "mask_ws.npy", mask)

        with TestClient(app) as c:
            sid = c.post(
                "/load", json={"filepath": str(tmp_path / "base_ws.npy")}
            ).json()["sid"]
            overlay_sid = c.post(
                "/load",
                json={"filepath": str(tmp_path / "mask_ws.npy"), "name": "overlay"},
            ).json()["sid"]
            with c.websocket_connect(f"/ws/{sid}") as ws:
                meta = ws.receive_json()
                assert meta["type"] == "metadata"

                ws.send_json(
                    {
                        "seq": 1,
                        "dim_x": 1,
                        "dim_y": 0,
                        "dim_z": -1,
                        "indices": [0, 0],
                        "colormap": "gray",
                        "dr": 0,
                        "complex_mode": 0,
                        "log_scale": False,
                        "slice_dim": 0,
                        "direction": 1,
                    }
                )
                plain_payload = ws.receive_bytes()

                ws.send_json(
                    {
                        "seq": 2,
                        "dim_x": 1,
                        "dim_y": 0,
                        "dim_z": -1,
                        "indices": [0, 0],
                        "colormap": "gray",
                        "dr": 0,
                        "complex_mode": 0,
                        "log_scale": False,
                        "slice_dim": 0,
                        "direction": 1,
                        "overlay_sid": overlay_sid,
                    }
                )
                overlay_payload = ws.receive_bytes()

        plain_rgba = np.frombuffer(plain_payload[20:], dtype=np.uint8).reshape(6, 6, 4)
        overlay_rgba = np.frombuffer(overlay_payload[20:], dtype=np.uint8).reshape(6, 6, 4)
        assert np.array_equal(plain_rgba[0, 0], overlay_rgba[0, 0])
        assert not np.array_equal(plain_rgba[2, 2], overlay_rgba[2, 2])
        assert overlay_rgba[2, 2, 0] > overlay_rgba[2, 2, 1]


# ---------------------------------------------------------------------------
# /grid
# ---------------------------------------------------------------------------


class TestGrid:
    def test_3d_returns_png(self, client, sid_3d):
        r = client.get(
            f"/grid/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "slice_dim": 0,
                "colormap": "gray",
                "dr": 0,
            },
        )
        assert r.status_code == 200
        assert "image/png" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content))
        # Mosaic of 20 frames of 64×64 → some reasonable size
        assert img.size[0] > 64 and img.size[1] > 64

    def test_unknown_sid_is_404(self, client):
        r = client.get(
            "/grid/doesnotexist000",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "slice_dim": 0,
                "colormap": "gray",
                "dr": 0,
            },
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /gif
# ---------------------------------------------------------------------------


class TestGif:
    def test_3d_returns_gif(self, client, sid_3d):
        r = client.get(
            f"/gif/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "slice_dim": 0,
                "colormap": "gray",
                "dr": 0,
            },
        )
        assert r.status_code == 200
        assert "image/gif" in r.headers["content-type"]
        img = Image.open(io.BytesIO(r.content))
        assert img.n_frames == 20  # one frame per slice-dim index


# ---------------------------------------------------------------------------
# /pixel
# ---------------------------------------------------------------------------


class TestPixel:
    def test_returns_numeric_value(self, client, sid_2d):
        r = client.get(
            f"/pixel/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 40,
                "py": 50,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "value" in body
        assert isinstance(body["value"], (int, float))

    def test_first_pixel_near_zero(self, client, sid_2d):
        """linspace(0,1, 100×80)[0,0] = 0.0."""
        r = client.get(
            f"/pixel/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 0,
                "py": 0,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        assert abs(r.json()["value"]) < 0.01

    def test_last_pixel_near_one(self, client, sid_2d):
        """linspace(0,1, 100×80)[-1,-1] ≈ 1.0."""
        r = client.get(
            f"/pixel/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 79,
                "py": 99,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        assert abs(r.json()["value"] - 1.0) < 0.01

    def test_unknown_sid_is_404(self, client):
        r = client.get(
            "/pixel/doesnotexist000",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 0,
                "py": 0,
            },
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /clearcache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clear_returns_ok(self, client, sid_2d):
        r = client.get(f"/clearcache/{sid_2d}")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_slice_still_works_after_clear(self, client, sid_2d):
        client.get(f"/clearcache/{sid_2d}")
        r = client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# /reload and /data_version  (--watch mode)
# ---------------------------------------------------------------------------


class TestReload:
    def test_data_version_initial_is_zero(self, client, sid_2d):
        r = client.get(f"/data_version/{sid_2d}")
        assert r.status_code == 200
        assert r.json()["version"] == 0

    def test_reload_bumps_data_version(self, client, tmp_path):
        """POST /reload/{sid} should increment data_version and return it."""
        arr = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        path = tmp_path / "watch.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]

        assert client.get(f"/data_version/{sid}").json()["version"] == 0

        # Overwrite file and reload
        arr2 = np.zeros((32, 32), dtype=np.float32)
        np.save(path, arr2)
        r = client.post(f"/reload/{sid}")
        assert r.status_code == 200
        assert r.json()["version"] == 1
        assert client.get(f"/data_version/{sid}").json()["version"] == 1

    def test_reload_unknown_sid_is_404(self, client):
        r = client.post("/reload/doesnotexist000")
        assert r.status_code == 404

    def test_reload_updates_shape(self, client, tmp_path):
        """After reload with a differently-shaped file, shape in metadata updates."""
        arr = np.zeros((8, 8), dtype=np.float32)
        path = tmp_path / "resized.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        assert client.get(f"/metadata/{sid}").json()["shape"] == [8, 8]

        # Save a larger array and reload
        np.save(path, np.zeros((16, 16), dtype=np.float32))
        client.post(f"/reload/{sid}")
        assert client.get(f"/metadata/{sid}").json()["shape"] == [16, 16]


# ---------------------------------------------------------------------------
# /update — ViewHandle.update() live-update endpoint
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_bumps_data_version(self, client, sid_2d):
        """POST /update/{sid} with .npy bytes should bump data_version."""
        buf = io.BytesIO()
        np.save(buf, np.zeros((10, 10), dtype=np.float32))
        r = client.post(f"/update/{sid_2d}", content=buf.getvalue())
        assert r.status_code == 200
        assert r.json()["version"] == 1

    def test_update_changes_shape(self, client, sid_2d):
        """After /update, metadata should reflect the new array shape."""
        new_arr = np.ones((5, 7, 3), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, new_arr)
        client.post(f"/update/{sid_2d}", content=buf.getvalue())
        meta = client.get(f"/metadata/{sid_2d}").json()
        assert meta["shape"] == [5, 7, 3]

    def test_update_unknown_sid_is_404(self, client):
        r = client.post("/update/doesnotexist000", content=b"")
        assert r.status_code == 404

    def test_update_bad_bytes_is_400(self, client, sid_2d):
        r = client.post(f"/update/{sid_2d}", content=b"not a numpy array")
        assert r.status_code == 400

    def test_update_clears_cache_and_recomputes_stats(self, client, sid_2d):
        """After update, data_version reflects sequential updates."""
        ones = np.ones((8, 8), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, ones)
        r1 = client.post(f"/update/{sid_2d}", content=buf.getvalue())
        v1 = r1.json()["version"]

        twos = np.full((8, 8), 2.0, dtype=np.float32)
        buf2 = io.BytesIO()
        np.save(buf2, twos)
        r2 = client.post(f"/update/{sid_2d}", content=buf2.getvalue())
        assert r2.json()["version"] == v1 + 1


class TestViewHandle:
    """Unit tests for the ViewHandle class (no server required)."""

    def test_view_handle_is_string_subclass(self):
        from arrayview._launcher import ViewHandle

        h = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)
        assert isinstance(h, str)
        assert str(h) == "http://localhost:8123/?sid=abc"

    def test_view_handle_properties(self):
        from arrayview._launcher import ViewHandle

        h = ViewHandle("http://localhost:9000/?sid=xyz", "xyz", 9000)
        assert h.sid == "xyz"
        assert h.port == 9000
        assert h.url == "http://localhost:9000/?sid=xyz"

    def test_view_handle_equality_with_str(self):
        from arrayview._launcher import ViewHandle

        h = ViewHandle("http://localhost:8123/?sid=abc", "abc", 8123)
        assert h == "http://localhost:8123/?sid=abc"


class TestLauncherShutdownWaiters:
    def test_cli_daemon_does_not_keep_post_close_idle(self):
        import arrayview._launcher as launcher

        assert launcher._CLI_DAEMON_IDLE_SECONDS == 0.0

    def test_wait_for_viewer_close_times_out_without_viewer(self, monkeypatch):
        import arrayview._launcher as launcher
        import arrayview._session as session_mod

        monkeypatch.setattr(session_mod, "VIEWER_SOCKETS", 0)

        start = time.monotonic()
        launcher._wait_for_viewer_close(connect_timeout=0.05)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5

    def test_wait_for_viewer_close_returns_after_disconnect(self, monkeypatch):
        import arrayview._launcher as launcher
        import arrayview._session as session_mod

        monkeypatch.setattr(session_mod, "VIEWER_SOCKETS", 0)

        def _simulate_viewer_lifecycle():
            time.sleep(0.05)
            session_mod.VIEWER_SOCKETS = 1
            time.sleep(0.35)
            session_mod.VIEWER_SOCKETS = 0

        worker = threading.Thread(target=_simulate_viewer_lifecycle, daemon=True)
        worker.start()

        start = time.monotonic()
        launcher._wait_for_viewer_close(
            grace_seconds=0.02,
            connect_timeout=1.0,
        )
        elapsed = time.monotonic() - start
        worker.join(timeout=1.0)

        assert elapsed < 1.0
        assert session_mod.VIEWER_SOCKETS == 0


class TestLauncherUrlHelpers:
    """Pure URL helper tests; no server required."""

    def test_viewer_url_preserves_existing_query_shape(self):
        from arrayview._launcher import _viewer_url

        assert _viewer_url(
            8123,
            "base",
            compare_sids=["cmp1", "cmp2"],
            overlay_sids=["ov1", "ov2"],
            overlay_colors=["ff4444", "44cc44"],
            dims=(0, 2),
            inline=True,
        ) == (
            "http://localhost:8123/?sid=base"
            "&overlay_sid=ov1,ov2"
            "&overlay_colors=ff4444,44cc44"
            "&compare_sid=cmp1"
            "&compare_sids=cmp1,cmp2"
            "&dim_x=0"
            "&dim_y=2"
            "&inline=1"
        )

    def test_viewer_path_matches_browser_url_path(self):
        from arrayview._launcher import _viewer_path

        assert _viewer_path("base", compare_sids=["cmp"]) == (
            "/?sid=base&compare_sid=cmp&compare_sids=cmp"
        )

    def test_shell_url_uses_localhost_and_existing_name_encoding(self):
        from arrayview._launcher import _shell_url

        assert _shell_url(
            9000, "base", "sample volume", compare_sids=["cmp1", "cmp2"]
        ) == (
            "http://localhost:9000/shell?init_sid=base"
            "&init_name=sample%20volume"
            "&init_compare_sid=cmp1"
            "&init_compare_sids=cmp1,cmp2"
        )


class TestLauncherDecisionHelpers:
    def test_resolve_cli_window_mode_prefers_browser_flag(self):
        from arrayview._launcher import _resolve_cli_window_mode

        plan = _resolve_cli_window_mode(
            explicit_window=None,
            browser_flag=True,
            config_window="native",
            in_vscode_terminal=True,
            is_vscode_remote=False,
            can_native_window=True,
        )

        assert plan == {
            "window_mode": "browser",
            "use_webview": False,
            "force_vscode": False,
            "requires_vscode_terminal": False,
            "warn_native_to_vscode": False,
        }

    def test_resolve_cli_window_mode_promotes_native_to_vscode_on_remote(self):
        from arrayview._launcher import _resolve_cli_window_mode

        plan = _resolve_cli_window_mode(
            explicit_window="native",
            browser_flag=False,
            config_window=None,
            in_vscode_terminal=True,
            is_vscode_remote=True,
            can_native_window=True,
        )

        assert plan == {
            "window_mode": "vscode",
            "use_webview": False,
            "force_vscode": True,
            "requires_vscode_terminal": False,
            "warn_native_to_vscode": True,
        }

    def test_resolve_cli_window_mode_uses_native_when_available(self):
        from arrayview._launcher import _resolve_cli_window_mode

        plan = _resolve_cli_window_mode(
            explicit_window=None,
            browser_flag=False,
            config_window=None,
            in_vscode_terminal=False,
            is_vscode_remote=False,
            can_native_window=True,
        )

        assert plan == {
            "window_mode": None,
            "use_webview": True,
            "force_vscode": False,
            "requires_vscode_terminal": False,
            "warn_native_to_vscode": False,
        }

    def test_should_notify_webview_disables_overlay_path(self):
        from arrayview._launcher import _should_notify_webview

        assert _should_notify_webview(True, None) is True
        assert _should_notify_webview(True, "sid_overlay") is False
        assert _should_notify_webview(False, None) is False

    def test_plan_cli_port_strategy_for_busy_ssh_tunnel(self):
        from arrayview._launcher import _plan_cli_port_strategy

        assert _plan_cli_port_strategy(
            port_in_use=True,
            is_arrayview_server=False,
            is_ssh=True,
            is_vscode_remote=False,
        ) == {
            "attempt_ssh_relay_before_scan": True,
            "requires_fixed_remote_port_error": False,
            "should_scan_for_port": True,
            "should_check_existing_ssh_relay": False,
        }

    def test_plan_cli_port_strategy_for_remote_busy_port(self):
        from arrayview._launcher import _plan_cli_port_strategy

        assert _plan_cli_port_strategy(
            port_in_use=True,
            is_arrayview_server=False,
            is_ssh=False,
            is_vscode_remote=True,
        ) == {
            "attempt_ssh_relay_before_scan": False,
            "requires_fixed_remote_port_error": True,
            "should_scan_for_port": False,
            "should_check_existing_ssh_relay": False,
        }

    def test_plan_cli_port_strategy_for_existing_ssh_relay(self):
        from arrayview._launcher import _plan_cli_port_strategy

        assert _plan_cli_port_strategy(
            port_in_use=False,
            is_arrayview_server=True,
            is_ssh=True,
            is_vscode_remote=False,
        ) == {
            "attempt_ssh_relay_before_scan": False,
            "requires_fixed_remote_port_error": False,
            "should_scan_for_port": False,
            "should_check_existing_ssh_relay": True,
        }

    def test_resolve_view_port_prefers_cli_server_on_remote(self):
        from arrayview._launcher import _resolve_view_port

        assert _resolve_view_port(
            8123, is_vscode_remote=True, cli_default_port_alive=True
        ) == 8000
        assert _resolve_view_port(
            8123, is_vscode_remote=False, cli_default_port_alive=True
        ) == 8123
        assert _resolve_view_port(
            9000, is_vscode_remote=True, cli_default_port_alive=True
        ) == 9000

    def test_select_arrayview_launch_path(self):
        from arrayview._launcher import _select_arrayview_launch_path

        assert (
            _select_arrayview_launch_path(
                is_arrayview_server=True, is_vscode_remote=True
            )
            == "existing_server"
        )
        assert (
            _select_arrayview_launch_path(
                is_arrayview_server=False, is_vscode_remote=True
            )
            == "remote_direct"
        )
        assert (
            _select_arrayview_launch_path(
                is_arrayview_server=False, is_vscode_remote=False
            )
            == "spawn_daemon"
        )


class TestLauncherDimsHelpers:
    def test_parse_dims_spec_accepts_xy_markers(self):
        from arrayview._launcher import _parse_dims_spec

        assert _parse_dims_spec("x,y,:,:") == (0, 1)
        assert _parse_dims_spec(":,:,y,x") == (3, 2)

    def test_parse_dims_spec_accepts_integer_pair(self):
        from arrayview._launcher import _parse_dims_spec

        assert _parse_dims_spec("2,3") == (2, 3)

    def test_parse_dims_spec_rejects_invalid_specs(self):
        from arrayview._launcher import _parse_dims_spec

        assert _parse_dims_spec("x,x,:,:") is None
        assert _parse_dims_spec("2,2") is None
        assert _parse_dims_spec("abc") is None


class TestViewWindowHelpers:
    def test_normalize_view_window_request_inline(self):
        from arrayview._launcher import _normalize_view_window_request

        assert _normalize_view_window_request("inline", None) == {
            "window": False,
            "inline": True,
            "force_browser": False,
            "force_vscode": False,
            "explicit_inline": False,
            "explicit_window": True,
        }

    def test_normalize_view_window_request_browser(self):
        from arrayview._launcher import _normalize_view_window_request

        assert _normalize_view_window_request("browser", True) == {
            "window": False,
            "inline": False,
            "force_browser": True,
            "force_vscode": False,
            "explicit_inline": True,
            "explicit_window": True,
        }

    def test_normalize_view_window_request_invalid_mode(self):
        from arrayview._launcher import _normalize_view_window_request

        with pytest.raises(ValueError, match="window must be"):
            _normalize_view_window_request("sideways", None)

    def test_resolve_view_display_defaults_applies_jupyter_and_config(self):
        from arrayview._launcher import _resolve_view_display_defaults

        assert _resolve_view_display_defaults(
            inline=None,
            window=None,
            is_jupyter=True,
            explicit_window=False,
            explicit_inline=False,
            force_browser=False,
            force_vscode=False,
            config_window=None,
        ) == {
            "inline": True,
            "window": False,
            "force_browser": False,
            "force_vscode": False,
        }

        assert _resolve_view_display_defaults(
            inline=None,
            window=None,
            is_jupyter=False,
            explicit_window=False,
            explicit_inline=False,
            force_browser=False,
            force_vscode=False,
            config_window="browser",
        ) == {
            "inline": False,
            "window": False,
            "force_browser": True,
            "force_vscode": False,
        }

    def test_promote_view_to_vscode_terminal_only_when_implicit(self):
        from arrayview._launcher import _promote_view_to_vscode_terminal

        assert _promote_view_to_vscode_terminal(
            in_vscode_terminal=True,
            inline=False,
            window=True,
            explicit_window=False,
            explicit_inline=False,
            force_vscode=False,
            force_browser=False,
        ) == {
            "window": False,
            "force_vscode": True,
        }

        assert _promote_view_to_vscode_terminal(
            in_vscode_terminal=True,
            inline=False,
            window=True,
            explicit_window=True,
            explicit_inline=False,
            force_vscode=False,
            force_browser=False,
        ) == {
            "window": True,
            "force_vscode": False,
        }


class TestCliOpenHelpers:
    def test_open_webview_cli_returns_after_ready_marker(self, monkeypatch):
        import arrayview._launcher as launcher

        calls = []

        class _DummyProc:
            pid = 12345
            returncode = None
            stderr = io.BytesIO()

            def poll(self):
                return None

        def _fake_open_webview(*args, ready_file=None, **kwargs):
            calls.append({"args": args, "ready_file": ready_file, **kwargs})
            with open(ready_file, "w") as f:
                f.write("ready")
            return _DummyProc()

        monkeypatch.setattr(launcher, "_open_webview", _fake_open_webview)

        assert launcher._open_webview_cli("http://localhost:8000/shell", 1200, 800)
        assert calls
        assert calls[0]["ready_file"]
        assert calls[0]["capture_stderr"] is True

    def test_open_webview_cli_returns_false_on_child_crash(self, monkeypatch):
        import arrayview._launcher as launcher

        class _DummyProc:
            pid = 12345
            returncode = 1
            stderr = io.BytesIO(b"boom")

            def poll(self):
                return self.returncode

        monkeypatch.setattr(launcher, "_open_webview", lambda *args, **kwargs: _DummyProc())

        assert not launcher._open_webview_cli("http://localhost:8000/shell", 1200, 800)

    def test_open_cli_existing_server_view_skips_open_when_already_notified(
        self, monkeypatch
    ):
        import arrayview._launcher as launcher

        calls = []
        monkeypatch.setattr(launcher, "_vprint", lambda *args, **kwargs: calls.append(args))
        monkeypatch.setattr(
            launcher,
            "_open_webview_cli",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        )
        monkeypatch.setattr(
            launcher,
            "_open_browser",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        )

        launcher._open_cli_existing_server_view(
            port=8000,
            sid="sid_base",
            compare_sids=["sid_cmp"],
            overlay_sid=None,
            dims_override=None,
            notify_webview=True,
            notified=True,
            name="base.npy",
            base_file="/tmp/base.npy",
            watch=False,
            window_mode=None,
            floating=False,
        )

        assert any("Injected into existing window" in args[0] for args in calls)

    def test_open_cli_existing_server_view_falls_back_to_browser(self, monkeypatch):
        import arrayview._launcher as launcher

        opened = []
        printed = []
        monkeypatch.setattr(launcher, "_open_webview_cli", lambda *args, **kwargs: False)
        monkeypatch.setattr(
            launcher, "_print_viewer_location", lambda url: printed.append(url)
        )
        monkeypatch.setattr(
            launcher,
            "_open_browser",
            lambda url, **kwargs: opened.append({"url": url, **kwargs}),
        )
        monkeypatch.setattr(launcher, "_vprint", lambda *args, **kwargs: None)

        launcher._open_cli_existing_server_view(
            port=8000,
            sid="sid_base",
            compare_sids=["sid_cmp"],
            overlay_sid=None,
            dims_override=(1, 2),
            notify_webview=True,
            notified=False,
            name="base.npy",
            base_file="/tmp/base.npy",
            watch=False,
            window_mode="native",
            floating=False,
        )

        assert printed == [
            "http://localhost:8000/?sid=sid_base&compare_sid=sid_cmp&compare_sids=sid_cmp&dim_x=1&dim_y=2"
        ]
        assert opened == [
            {
                "url": "http://localhost:8000/?sid=sid_base&compare_sid=sid_cmp&compare_sids=sid_cmp&dim_x=1&dim_y=2",
                "blocking": True,
                "title": "ArrayView: base.npy",
                "filepath": "/tmp/base.npy",
                "floating": False,
            }
        ]

    def test_open_cli_spawned_view_overlay_browser_path_starts_watch(self, monkeypatch):
        import arrayview._launcher as launcher

        opened = []
        printed = []
        watched = []
        monkeypatch.setattr(
            launcher,
            "_open_webview_cli",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        )
        monkeypatch.setattr(
            launcher, "_print_viewer_location", lambda url: printed.append(url)
        )
        monkeypatch.setattr(
            launcher,
            "_start_watch_thread",
            lambda filepath, sid, port: watched.append((filepath, sid, port)),
        )
        monkeypatch.setattr(
            launcher,
            "_open_browser",
            lambda url, **kwargs: opened.append({"url": url, **kwargs}),
        )
        monkeypatch.setattr(launcher, "_vprint", lambda *args, **kwargs: None)

        launcher._open_cli_spawned_view(
            port=8000,
            sid="sid_base",
            compare_sids=[],
            overlay_sid="sid_overlay",
            dims_override=None,
            use_webview=True,
            name="base.npy",
            watch=True,
            window_mode="browser",
            floating=True,
            is_remote=False,
            base_file="/tmp/base.npy",
        )

        assert printed == ["http://localhost:8000/?sid=sid_base&overlay_sid=sid_overlay"]
        assert watched == [("/tmp/base.npy", "sid_base", 8000)]
        assert opened == [
            {
                "url": "http://localhost:8000/?sid=sid_base&overlay_sid=sid_overlay",
                "blocking": True,
                "force_vscode": False,
                "title": "ArrayView: base.npy",
                "floating": True,
            }
        ]

    def test_register_cli_session_with_existing_server_returns_open_state(
        self, monkeypatch
    ):
        import arrayview._launcher as launcher

        attach_calls = []

        def fake_load(port, filepath, name, **kwargs):
            if filepath.endswith("overlay.npy"):
                return {"sid": "sid_overlay", "name": name}
            if filepath.endswith("cmp.npy"):
                return {"sid": "sid_cmp", "name": name}
            return {"sid": "sid_base", "name": name, "notified": True}

        monkeypatch.setattr(launcher, "_load_session_from_filepath", fake_load)
        monkeypatch.setattr(
            launcher,
            "_attach_vectorfield_to_session",
            lambda port, sid, filepath, components_dim=None: attach_calls.append(
                (port, sid, filepath, components_dim)
            )
            or {"ok": True},
        )

        result = launcher._register_cli_session_with_existing_server(
            port=8000,
            overlay_paths=["/tmp/overlay.npy"],
            compare_files=["/tmp/cmp.npy"],
            base_file="/tmp/base.npy",
            name="base.npy",
            rgb=True,
            use_webview=True,
            vectorfield="/tmp/vf.npy",
            vfield_components_dim=2,
        )

        assert result == {
            "sid": "sid_base",
            "overlay_sid": "sid_overlay",
            "compare_sids": ["sid_cmp"],
            "notify_webview": False,
            "notified": True,
        }
        assert attach_calls == [(8000, "sid_base", "/tmp/vf.npy", 2)]

    def test_handle_cli_existing_server_opens_registered_session(self, monkeypatch):
        import arrayview._launcher as launcher

        opened = []
        monkeypatch.setattr(
            launcher,
            "_register_cli_session_with_existing_server",
            lambda **kwargs: {
                "sid": "sid_base",
                "overlay_sid": None,
                "compare_sids": ["sid_cmp"],
                "notify_webview": True,
                "notified": False,
            },
        )
        monkeypatch.setattr(
            launcher,
            "_open_cli_existing_server_view",
            lambda **kwargs: opened.append(kwargs),
        )

        launcher._handle_cli_existing_server(
            port=8000,
            base_file="/tmp/base.npy",
            name="base.npy",
            compare_files=["/tmp/cmp.npy"],
            overlay_files=[],
            rgb=False,
            vectorfield=None,
            vfield_components_dim=None,
            use_webview=True,
            dims_override=(1, 2),
            watch=True,
            window_mode="native",
            floating=False,
        )

        assert opened == [
            {
                "port": 8000,
                "sid": "sid_base",
                "compare_sids": ["sid_cmp"],
                "overlay_sid": None,
                "dims_override": (1, 2),
                "notify_webview": True,
                "notified": False,
                "name": "base.npy",
                "base_file": "/tmp/base.npy",
                "watch": True,
                "window_mode": "native",
                "floating": False,
            }
        ]

    def test_handle_cli_spawned_daemon_opens_spawned_session(self, monkeypatch):
        import arrayview._launcher as launcher

        spawned = []
        opened = []

        monkeypatch.setattr(launcher, "_configure_vscode_port_preview", lambda port: None)
        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            lambda cmd: spawned.append(cmd) or object(),
        )
        monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
        monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: ["sid_cmp"])
        monkeypatch.setattr(
            launcher,
            "_open_cli_spawned_view",
            lambda **kwargs: opened.append(kwargs),
        )

        monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid_base"})())
        launcher._handle_cli_spawned_daemon(
            port=8000,
            base_file="/tmp/base.npy",
            name="base.npy",
            compare_files=["/tmp/cmp.npy"],
            overlay_files=["/tmp/overlay.npy"],
            dims_override=(1, 2),
            use_webview=False,
            watch=True,
            window_mode="browser",
            floating=False,
            is_remote=False,
            vectorfield="/tmp/vf.npy",
            vfield_components_dim=2,
            rgb=True,
            demo_name="demo",
            demo_cleanup=True,
        )

        assert spawned
        assert "_serve_daemon(" in spawned[0][2]
        assert "persist=False" in spawned[0][2]
        assert "rgb=True" in spawned[0][2]
        assert opened == [
            {
                "port": 8000,
                "sid": "sid_base",
                "compare_sids": ["sid_cmp"],
                "overlay_sid": "sid_base",
                "dims_override": (1, 2),
                "use_webview": False,
                "name": "base.npy",
                "base_file": "/tmp/base.npy",
                "watch": True,
                "window_mode": "browser",
                "floating": False,
                "is_remote": False,
                "webview_already_opened": False,
            }
        ]

    def test_handle_cli_spawned_daemon_opens_native_shell_before_port_wait(
        self, monkeypatch
    ):
        import arrayview._launcher as launcher

        events = []
        opened = []

        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            lambda cmd: events.append(("spawn", cmd)) or object(),
        )
        monkeypatch.setattr(
            launcher,
            "_open_webview_cli",
            lambda *args, **kwargs: events.append(("webview", args, kwargs)) or True,
        )
        monkeypatch.setattr(
            launcher,
            "_wait_for_port",
            lambda *args, **kwargs: events.append(("wait", args, kwargs)) or True,
        )
        monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
        monkeypatch.setattr(
            launcher,
            "_notify_existing_session",
            lambda *args, **kwargs: events.append(("notify", args, kwargs))
            or {"notified": True},
        )
        monkeypatch.setattr(
            launcher,
            "_open_cli_spawned_view",
            lambda **kwargs: opened.append(kwargs),
        )
        monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid_base"})())

        launcher._handle_cli_spawned_daemon(
            port=8000,
            base_file="/tmp/base.npy",
            name="base.npy",
            compare_files=[],
            overlay_files=[],
            dims_override=None,
            use_webview=True,
            watch=False,
            window_mode="native",
            floating=False,
            is_remote=False,
            vectorfield=None,
            vfield_components_dim=None,
            rgb=False,
            demo_name=None,
            demo_cleanup=False,
        )

        event_names = [event[0] for event in events]
        assert event_names[:3] == ["spawn", "webview", "wait"]
        assert "notify" in event_names
        assert events[1][2]["shell_port"] == 8000
        assert opened[0]["webview_already_opened"] is True

    def test_handle_cli_spawned_daemon_does_not_early_open_remote_webview(
        self, monkeypatch
    ):
        import arrayview._launcher as launcher

        events = []

        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            lambda cmd: events.append(("spawn", cmd)) or object(),
        )
        monkeypatch.setattr(
            launcher,
            "_open_webview_cli",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
        )
        monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
        monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
        monkeypatch.setattr(
            launcher,
            "_open_cli_spawned_view",
            lambda **kwargs: events.append(("open", kwargs)),
        )
        monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid_base"})())

        launcher._handle_cli_spawned_daemon(
            port=8000,
            base_file="/tmp/base.npy",
            name="base.npy",
            compare_files=[],
            overlay_files=[],
            dims_override=None,
            use_webview=True,
            watch=False,
            window_mode="native",
            floating=False,
            is_remote=True,
            vectorfield=None,
            vfield_components_dim=None,
            rgb=False,
            demo_name=None,
            demo_cleanup=False,
        )

        assert events[0][0] == "spawn"
        assert events[1][0] == "open"
        assert events[1][1]["webview_already_opened"] is False


# ---------------------------------------------------------------------------
# /histogram — histogram strip endpoint
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_histogram_returns_counts_and_edges(self, client, sid_2d):
        r = client.get(
            f"/histogram/{sid_2d}",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "counts" in body
        assert "edges" in body
        assert "vmin" in body
        assert "vmax" in body
        assert len(body["counts"]) >= 8
        assert len(body["edges"]) == len(body["counts"]) + 1

    def test_histogram_unknown_sid_is_404(self, client):
        r = client.get(
            "/histogram/doesnotexist000",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0"},
        )
        assert r.status_code == 404

    def test_histogram_bin_count_respected(self, client, sid_2d):
        r = client.get(
            f"/histogram/{sid_2d}",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0", "bins": 32},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["counts"]) == 32


# ---------------------------------------------------------------------------
# Volume histogram (scroll-dim subsampled)
# ---------------------------------------------------------------------------


class TestVolumeHistogram:
    """Tests for /volume-histogram/{sid} endpoint."""

    def test_volume_histogram_returns_counts_and_edges(self, client, sid_3d):
        """3D array: sample along dim 2 (scroll), display dims 0,1."""
        resp = client.get(f"/volume-histogram/{sid_3d}", params={
            "dim_x": 0, "dim_y": 1, "scroll_dim": 2, "bins": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "counts" in data and "edges" in data
        assert len(data["counts"]) == 32
        assert len(data["edges"]) == 33
        assert "vmin" in data and "vmax" in data
        assert data["vmin"] < data["vmax"]

    def test_volume_histogram_with_fixed_indices(self, client, sid_4d):
        """4D array: fix dim 0 (parameter map), scroll along dim 1, display dims 2,3."""
        resp = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:0", "bins": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["counts"]) == 32

    def test_volume_histogram_different_fixed_indices_differ(self, client, sid_4d):
        """Different fixed indices should give different histograms (different parameter maps)."""
        resp_a = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:0", "bins": 32,
        })
        resp_b = client.get(f"/volume-histogram/{sid_4d}", params={
            "dim_x": 2, "dim_y": 3, "scroll_dim": 1,
            "fixed_indices": "0:2", "bins": 32,
        })
        a = resp_a.json()
        b = resp_b.json()
        assert a["counts"] != b["counts"] or a["vmin"] != b["vmin"]

    def test_volume_histogram_unknown_sid_is_404(self, client):
        resp = client.get("/volume-histogram/nonexistent", params={
            "dim_x": 0, "dim_y": 1, "scroll_dim": 2,
        })
        assert resp.status_code == 404

    def test_volume_histogram_caches_result(self, client, sid_3d):
        """Two identical requests should return identical results (cache hit)."""
        params = {"dim_x": 0, "dim_y": 1, "scroll_dim": 2, "bins": 32}
        resp1 = client.get(f"/volume-histogram/{sid_3d}", params=params)
        resp2 = client.get(f"/volume-histogram/{sid_3d}", params=params)
        assert resp1.json() == resp2.json()


# ---------------------------------------------------------------------------
# Memory-aware cache (byte limits)
# ---------------------------------------------------------------------------


class TestROI:
    def test_roi_returns_stats(self, client, sid_2d):
        # arr_2d is linspace(0,1) shaped 100×80; request a region we know
        r = client.get(
            f"/roi/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "x0": 0,
                "y0": 0,
                "x1": 10,
                "y1": 10,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert (
            "min" in body
            and "max" in body
            and "mean" in body
            and "std" in body
            and "n" in body
        )
        assert body["n"] == 121  # 11×11 pixels
        assert body["min"] <= body["mean"] <= body["max"]

    def test_roi_unknown_sid_is_404(self, client):
        r = client.get(
            "/roi/doesnotexist000",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "x0": 0,
                "y0": 0,
                "x1": 5,
                "y1": 5,
            },
        )
        assert r.status_code == 404


class TestFloodFillROI:
    def test_floodfill_returns_stats_and_mask(self, client, sid_2d):
        r = client.get(
            f"/roi_floodfill/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 40,
                "py": 50,
                "tolerance": 0.05,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "min" in body and "max" in body and "mean" in body
        assert "n" in body and body["n"] > 0
        assert "seed_value" in body
        assert "bbox" in body
        assert "mask_b64" in body

    def test_floodfill_unknown_sid_is_404(self, client):
        r = client.get(
            "/roi_floodfill/doesnotexist000",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "px": 0,
                "py": 0,
            },
        )
        assert r.status_code == 404

    def test_floodfill_larger_tolerance_gives_more_pixels(self, client, sid_2d):
        base = {
            "dim_x": 1,
            "dim_y": 0,
            "indices": "0,0",
            "px": 40,
            "py": 50,
            "complex_mode": 0,
        }
        r_small = client.get(
            f"/roi_floodfill/{sid_2d}", params={**base, "tolerance": 0.01}
        )
        r_large = client.get(
            f"/roi_floodfill/{sid_2d}", params={**base, "tolerance": 0.5}
        )
        assert r_small.status_code == 200
        assert r_large.status_code == 200
        n_small = r_small.json()["n"]
        n_large = r_large.json()["n"]
        assert n_large >= n_small


class TestColormap:
    def test_known_matplotlib_colormap_returns_200(self, client):
        r = client.get("/colormap/hot")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "gradient_stops" in body
        assert len(body["gradient_stops"]) > 0

    def test_unknown_colormap_returns_404(self, client):
        r = client.get("/colormap/definitely_not_a_real_colormap_xyz")
        assert r.status_code == 404

    def test_builtin_colormap_returns_200(self, client):
        r = client.get("/colormap/viridis")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------


class TestDiff:
    """Tests for the /diff/{sid_a}/{sid_b} compare-diff endpoint."""

    def _make_pair(self, client, tmp_path):
        arr_a = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        arr_b = np.linspace(1, 0, 64 * 64, dtype=np.float32).reshape(64, 64)
        np.save(tmp_path / "da.npy", arr_a)
        np.save(tmp_path / "db.npy", arr_b)
        sid_a = client.post("/load", json={"filepath": str(tmp_path / "da.npy")}).json()["sid"]
        sid_b = client.post("/load", json={"filepath": str(tmp_path / "db.npy")}).json()["sid"]
        return sid_a, sid_b

    def test_diff_mode1_returns_image_with_headers(self, client, tmp_path):
        sid_a, sid_b = self._make_pair(client, tmp_path)
        r = client.get(f"/diff/{sid_a}/{sid_b}", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0", "diff_mode": 1,
        })
        assert r.status_code == 200
        assert "image/jpeg" in r.headers["content-type"]
        assert r.headers.get("X-ArrayView-Colormap") == "RdBu_r"

    def test_diff_mode2_uses_afmhot(self, client, tmp_path):
        sid_a, sid_b = self._make_pair(client, tmp_path)
        r = client.get(f"/diff/{sid_a}/{sid_b}", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0", "diff_mode": 2,
        })
        assert r.status_code == 200
        assert r.headers.get("X-ArrayView-Colormap") == "afmhot"

    def test_diff_colormap_override(self, client, tmp_path):
        sid_a, sid_b = self._make_pair(client, tmp_path)
        r = client.get(f"/diff/{sid_a}/{sid_b}", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0", "diff_mode": 1,
            "diff_colormap": "viridis",
        })
        assert r.status_code == 200
        assert r.headers.get("X-ArrayView-Colormap") == "viridis"

    def test_unknown_sid_returns_404(self, client):
        r = client.get("/diff/nosid1/nosid2", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0", "diff_mode": 1,
        })
        assert r.status_code == 404


class TestMemoryAwareCache:
    def test_byte_counters_reset_on_clearcache(self, client, sid_2d):
        """After clearcache, byte counters should be 0."""
        from arrayview._app import SESSIONS

        # Warm the cache by requesting a slice
        client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        client.get(f"/clearcache/{sid_2d}")
        session = SESSIONS.get(sid_2d)
        if session:
            assert session._raw_bytes == 0
            assert session._rgba_bytes == 0
            assert session._mosaic_bytes == 0

    def test_byte_counters_positive_after_slice(self, client, sid_2d):
        """After rendering a slice, raw byte counter should be > 0."""
        from arrayview._app import SESSIONS

        client.get(f"/clearcache/{sid_2d}")
        client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "colormap": "gray",
                "dr": 0,
                "slice_dim": 0,
            },
        )
        session = SESSIONS.get(sid_2d)
        if session:
            assert session._raw_bytes > 0


# ---------------------------------------------------------------------------
# Port / tunnel helper tests
# ---------------------------------------------------------------------------
class TestPortAndTunnelHelpers:
    def test_find_server_port_returns_alive_port(self, server_url):
        """If an arrayview server is live, _find_server_port returns it as already_running."""
        import re
        from arrayview._app import _find_server_port

        port = int(re.search(r":(\d+)", server_url).group(1))
        found, alive = _find_server_port(port)
        assert alive is True
        assert found == port

    def test_find_server_port_avoids_busy_port(self):
        """When preferred port is busy (not arrayview), _find_server_port scans ahead."""
        import socket as _socket
        from arrayview._app import _find_server_port

        # Bind and listen to make the port genuinely appear in use.
        # Use a large backlog so _server_alive's probe doesn't consume the only slot.
        with _socket.socket() as s:
            s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(128)
            busy_port = s.getsockname()[1]
            found, alive = _find_server_port(busy_port, search_range=20)
            assert alive is False
            assert found != busy_port  # scanned past the busy port

    def test_in_vscode_tunnel_false_in_clean_env(self, monkeypatch):
        from arrayview._app import _in_vscode_tunnel

        for k in (
            "VSCODE_INJECTION",
            "VSCODE_AGENT_FOLDER",
            "SSH_CLIENT",
            "SSH_CONNECTION",
        ):
            monkeypatch.delenv(k, raising=False)
        assert _in_vscode_tunnel() is False

    def test_in_vscode_tunnel_true_with_ssh(self, monkeypatch):
        from arrayview._app import _in_vscode_tunnel

        monkeypatch.setenv("SSH_CLIENT", "127.0.0.1 12345 22")
        assert _in_vscode_tunnel() is True

    def test_can_native_window_false_in_tunnel(self, monkeypatch):
        from arrayview._app import _can_native_window

        monkeypatch.setenv("SSH_CLIENT", "127.0.0.1 12345 22")
        assert _can_native_window() is False

    def test_in_vscode_terminal_false_for_local_matlab(self, monkeypatch):
        import arrayview._platform as platform

        monkeypatch.setattr(platform, "_MATLAB_CACHE", None)
        monkeypatch.setattr(platform, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(platform, "_in_matlab", lambda: True)
        monkeypatch.setenv("TERM_PROGRAM", "vscode")
        monkeypatch.setattr(platform, "_find_vscode_ipc_hook", lambda: "/tmp/vscode-ipc")

        assert platform._in_vscode_terminal() is False

    def test_can_native_window_allows_local_matlab_despite_vscode_hook(self, monkeypatch):
        import arrayview._platform as platform

        monkeypatch.setattr(platform, "_in_vscode_terminal", lambda: False)
        monkeypatch.setattr(platform, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(platform, "_find_vscode_ipc_hook", lambda: "/tmp/vscode-ipc")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.setattr(platform.sys, "platform", "darwin")

        assert platform._can_native_window() is True

    def test_open_browser_skips_vscode_signal_when_terminal_check_is_false(self, monkeypatch):
        import arrayview._vscode_browser as browser_mod

        signal_calls = []
        open_calls = []

        monkeypatch.setattr(browser_mod, "_in_vscode_terminal", lambda: False)
        monkeypatch.setattr(browser_mod, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(browser_mod, "_open_via_signal_file", lambda *args, **kwargs: signal_calls.append(args))
        monkeypatch.setattr(browser_mod.sys, "platform", "darwin")
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_CONNECTION", raising=False)

        def _fake_run(cmd, capture_output=True, timeout=5):
            open_calls.append(cmd)
            return type("Completed", (), {"returncode": 0})()

        monkeypatch.setattr(browser_mod.subprocess, "run", _fake_run)

        browser_mod._open_browser("http://localhost:8123/?sid=sid_matlab", blocking=True)

        assert signal_calls == []
        assert open_calls == [["open", "http://localhost:8123/?sid=sid_matlab"]]


# ---------------------------------------------------------------------------
# Overlay heatmap: _overlay_is_label_map
# ---------------------------------------------------------------------------


class TestOverlayIsLabelMap:
    """_overlay_is_label_map returns True for small integer label maps only."""

    def _register(self, arr, tmp_path, client, name):
        from tests.conftest import register_array

        return register_array(client, arr, tmp_path, name)

    def test_float_array_is_not_label_map(self, client, tmp_path):
        from arrayview._render import _overlay_is_label_map
        from arrayview._app import SESSIONS, Session

        arr = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        s = Session(arr, name="ov_float")
        SESSIONS[s.sid] = s
        assert _overlay_is_label_map(s.sid) is False

    def test_integer_few_labels_is_label_map(self, client, tmp_path):
        from arrayview._render import _overlay_is_label_map
        from arrayview._app import SESSIONS, Session

        arr = np.zeros((32, 32), dtype=np.int32)
        arr[5:15, 5:15] = 1
        arr[15:25, 15:25] = 2
        s = Session(arr, name="ov_labels")
        SESSIONS[s.sid] = s
        assert _overlay_is_label_map(s.sid) is True

    def test_integer_many_unique_values_is_heatmap(self, client, tmp_path):
        from arrayview._render import _overlay_is_label_map
        from arrayview._app import SESSIONS, Session

        arr = np.arange(32 * 32, dtype=np.int32).reshape(32, 32)  # 1024 unique values
        s = Session(arr, name="ov_many")
        SESSIONS[s.sid] = s
        ov_raw = arr.astype(np.float32)
        assert _overlay_is_label_map(s.sid, ov_raw) is False

    def test_integer_exactly_16_labels_is_label_map(self, client, tmp_path):
        from arrayview._render import _overlay_is_label_map
        from arrayview._app import SESSIONS, Session

        arr = np.zeros((32, 32), dtype=np.int32)
        for i in range(16):
            arr[i * 2, :] = i + 1  # labels 1..16
        s = Session(arr, name="ov_16")
        SESSIONS[s.sid] = s
        ov_raw = arr.astype(np.float32)
        assert _overlay_is_label_map(s.sid, ov_raw) is True


# ---------------------------------------------------------------------------
# Drag-and-drop upload: /load-upload
# ---------------------------------------------------------------------------


class TestLoadUpload:
    def test_upload_npy_creates_session(self, client, tmp_path):
        arr = np.random.default_rng(1).standard_normal((20, 30)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)
        r = client.post(
            "/load-upload",
            files={"file": ("test_array.npy", buf, "application/octet-stream")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "sid" in body
        assert body["name"] == "test_array.npy"

    def test_upload_unsupported_type_returns_400(self, client):
        r = client.post(
            "/load-upload",
            files={"file": ("bad.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 400

    def test_upload_bad_npy_returns_400(self, client):
        r = client.post(
            "/load-upload",
            files={"file": ("bad.npy", b"not numpy data", "application/octet-stream")},
        )
        assert r.status_code == 400


class TestObliquePersistence:
    def test_oblique_save_and_load_recent_roundtrip(self, client, sid_3d, tmp_path, monkeypatch):
        import arrayview._routes_persistence as routes_persistence

        monkeypatch.setattr(
            routes_persistence,
            "_OBLIQUE_RECENT_FILE",
            str(tmp_path / "oblique_recent.json"),
        )

        preset = {
            "version": 1,
            "shape": [20, 64, 64],
            "mv_dims": [0, 1, 2],
            "indices": [10, 20, 30],
            "oblique_vecs": [
                {"bh": [1.0, 0.0, 0.0], "bv": [0.0, 1.0, 0.0], "n": [0.0, 0.0, 1.0]},
                {"bh": [0.0, 1.0, 0.0], "bv": [0.0, 0.0, 1.0], "n": [1.0, 0.0, 0.0]},
                {"bh": [1.0, 0.0, 0.0], "bv": [0.0, 0.0, 1.0], "n": [0.0, -1.0, 0.0]},
            ],
            "pane_labels": ["A", "B", "C"],
            "oblique_ortho_lock": True,
        }

        save = client.post("/oblique/save", json={"sid": sid_3d, "preset": preset})
        assert save.status_code == 200
        assert save.json()["preset"]["indices"] == [10, 20, 30]

        load = client.post("/oblique/load_recent", json={"sid": sid_3d})
        assert load.status_code == 200
        body = load.json()
        assert body["ok"] is True
        assert body["preset"]["mv_dims"] == [0, 1, 2]
        assert body["preset"]["pane_labels"] == ["A", "B", "C"]
        assert body["preset"]["oblique_ortho_lock"] is True


class TestCropPersistence:
    def test_crop_register_confirm_reload_and_clear(self, client, sid_3d, tmp_path):
        recent_file = tmp_path / "crop_recent.json"

        register = client.post(
            "/crop/register",
            json={"sid": sid_3d, "readout_dim": 2, "recent_file": str(recent_file)},
        )
        assert register.status_code == 200
        body = register.json()
        assert body["readout_dim"] == 2
        assert body["recent_file"] == str(recent_file)

        update = client.post(
            "/crop/update",
            json={"sid": sid_3d, "x_start": 3, "x_end": 11, "viz_z": 4, "viz_y": 5},
        )
        assert update.status_code == 200
        updated = update.json()
        assert updated["x_start"] == 3
        assert updated["x_end"] == 11
        assert updated["viz_z"] == 4
        assert updated["viz_y"] == 5

        confirm = client.post("/crop/confirm", json={"sid": sid_3d})
        assert confirm.status_code == 200
        confirmed = confirm.json()
        assert confirmed["confirmed"] is True
        assert confirmed["saved_recent"] is True
        assert confirmed["saved_recent_path"] == str(recent_file)

        clear = client.post("/crop/clear", json={"sid": sid_3d})
        assert clear.status_code == 200
        assert clear.json()["ok"] is True

        missing = client.get(f"/crop/state/{sid_3d}")
        assert missing.status_code == 404

        register_again = client.post(
            "/crop/register",
            json={"sid": sid_3d, "readout_dim": 2, "recent_file": str(recent_file)},
        )
        assert register_again.status_code == 200
        reloaded = register_again.json()
        assert reloaded["loaded_recent"] is True
        assert reloaded["x_start"] == 3
        assert reloaded["x_end"] == 11
        assert reloaded["viz_z"] == 4
        assert reloaded["viz_y"] == 5


# ---------------------------------------------------------------------------
# Multiple overlays: /slice with overlay_sid and overlay_colors
# ---------------------------------------------------------------------------


class TestMultipleOverlays:
    def test_single_overlay_with_color(self, client, sid_2d):
        """A single binary mask overlay with explicit hex color renders without error."""
        from arrayview._app import SESSIONS, Session

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        ov_session = Session(mask, name="mask1")
        SESSIONS[ov_session.sid] = ov_session

        r = client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "overlay_sid": ov_session.sid,
                "overlay_colors": "ff4444",
            },
        )
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width > 0 and img.height > 0

    def test_two_overlays_comma_separated(self, client, sid_2d):
        """Two overlays passed as comma-separated sids are composited without error."""
        from arrayview._app import SESSIONS, Session

        mask1 = np.zeros((64, 64), dtype=np.uint8)
        mask1[5:20, 5:20] = 1
        mask2 = np.zeros((64, 64), dtype=np.uint8)
        mask2[40:55, 40:55] = 1
        ov1 = Session(mask1, name="mask1")
        ov2 = Session(mask2, name="mask2")
        SESSIONS[ov1.sid] = ov1
        SESSIONS[ov2.sid] = ov2

        r = client.get(
            f"/slice/{sid_2d}",
            params={
                "dim_x": 1,
                "dim_y": 0,
                "indices": "0,0",
                "overlay_sid": f"{ov1.sid},{ov2.sid}",
                "overlay_colors": "ff4444,44cc44",
            },
        )
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.width > 0 and img.height > 0

    def test_composite_overlays_helper(self, client, sid_2d):
        """_composite_overlays helper iterates all sids and returns modified rgba."""
        from arrayview._server import _composite_overlays
        from arrayview._app import SESSIONS, Session

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        ov = Session(mask, name="ov")
        SESSIONS[ov.sid] = ov

        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        result = _composite_overlays(
            rgba,
            ov.sid,
            "ff0000",
            0.5,
            dim_x=1,
            dim_y=0,
            idx_tuple=(0, 0),
            shape_hw=(64, 64),
        )
        # The mask region should have been tinted red
        assert result[25, 25, 0] > 0  # red channel non-zero in masked area
        assert result[0, 0, 0] == 0  # outside mask unchanged


# ---------------------------------------------------------------------------
# /line_profile
# ---------------------------------------------------------------------------


class TestLineProfile:
    def test_line_profile_returns_values_and_distance(self, client, sid_3d):
        r = client.get(
            f"/line_profile/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "x0": 0,
                "y0": 0,
                "x1": 63,
                "y1": 63,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "values" in body
        assert "distance" in body
        assert isinstance(body["values"], list)
        assert isinstance(body["distance"], float)
        assert body["distance"] > 0

    def test_line_profile_values_length(self, client, sid_3d):
        r = client.get(
            f"/line_profile/{sid_3d}",
            params={
                "dim_x": 2,
                "dim_y": 1,
                "indices": "0,0,0",
                "x0": 0,
                "y0": 0,
                "x1": 30,
                "y1": 30,
                "complex_mode": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["values"]) == 200


# ---------------------------------------------------------------------------
# Complex-aware projections
# ---------------------------------------------------------------------------


@pytest.fixture
def complex_sid(client, tmp_path):
    """Load a 3D complex array for projection tests."""
    arr = np.array(
        [[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
         [[2 + 1j, 4 + 3j], [6 + 5j, 8 + 7j]]],
        dtype=np.complex64,
    )
    path = tmp_path / "complex.npy"
    np.save(str(path), arr)
    resp = client.post("/load", json={"filepath": str(path), "name": "complex_test"})
    return resp.json()["sid"]


class TestComplexProjections:
    """Projections on complex data must not trigger ComplexWarning."""

    def test_complex_projection_max(self, client, complex_sid):
        import warnings
        from numpy.exceptions import ComplexWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", ComplexWarning)
            resp = client.get(
                f"/slice/{complex_sid}",
                params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                        "projection_dim": 2, "projection_mode": 1},
            )
        assert resp.status_code == 200

    def test_complex_projection_min(self, client, complex_sid):
        import warnings
        from numpy.exceptions import ComplexWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", ComplexWarning)
            resp = client.get(
                f"/slice/{complex_sid}",
                params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                        "projection_dim": 2, "projection_mode": 2},
            )
        assert resp.status_code == 200

    def test_complex_projection_sos(self, client, complex_sid):
        import warnings
        from numpy.exceptions import ComplexWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", ComplexWarning)
            resp = client.get(
                f"/slice/{complex_sid}",
                params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                        "projection_dim": 2, "projection_mode": 5},
            )
        assert resp.status_code == 200

    def test_complex_projection_mean(self, client, complex_sid):
        import warnings
        from numpy.exceptions import ComplexWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", ComplexWarning)
            resp = client.get(
                f"/slice/{complex_sid}",
                params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                        "projection_dim": 2, "projection_mode": 3},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# view() validation guards (no server required)
# ---------------------------------------------------------------------------


class TestViewValidation:
    """Pure validation tests for view() — no server or display needed."""

    def test_zero_arrays_raises(self):
        from arrayview._launcher import view

        with pytest.raises(ValueError, match="at least one array"):
            view()

    def test_five_arrays_raises(self):
        from arrayview._launcher import view

        a = np.zeros((3, 3))
        with pytest.raises(ValueError, match="at most 4"):
            view(a, a, a, a, a)

    def test_mismatched_name_list_raises(self):
        from arrayview._launcher import view

        a = np.zeros((3, 3))
        with pytest.raises(ValueError, match="name list length"):
            view(a, a, name=["x"])

    def test_mismatched_rgb_list_raises(self):
        from arrayview._launcher import view

        a = np.zeros((3, 3))
        with pytest.raises(ValueError, match="rgb list length"):
            view(a, a, rgb=[True])


class TestViewDisplayRouting:
    def test_local_matlab_view_prefers_native_window(self, monkeypatch):
        import arrayview._launcher as launcher
        import arrayview._session as session_mod

        native_calls = []
        browser_calls = []

        class _DummyEvent:
            def clear(self):
                return None

            def wait(self, timeout=None):
                return True

        class _DummyThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target

            def start(self):
                return None

        class _DummyProc:
            def poll(self):
                return None

        monkeypatch.setattr(launcher, "_in_jupyter", lambda: False)
        monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: False)
        monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
        monkeypatch.setattr(launcher, "_can_native_window", lambda: True)
        monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
        monkeypatch.setattr(launcher, "_server_pid", lambda port: None)
        monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
        monkeypatch.setattr("arrayview._config.get_window_default", lambda _env: None)
        monkeypatch.setattr("arrayview._platform.detect_environment", lambda: "terminal")
        monkeypatch.setattr(launcher, "_server_ready_event", _DummyEvent())
        monkeypatch.setattr(launcher.threading, "Thread", _DummyThread)
        monkeypatch.setattr(
            launcher,
            "_open_webview_with_fallback",
            lambda url, *args, **kwargs: native_calls.append({"url": url, **kwargs}) or _DummyProc(),
        )
        monkeypatch.setattr(
            launcher,
            "_open_browser",
            lambda url, **kwargs: browser_calls.append({"url": url, **kwargs}),
        )
        monkeypatch.setattr(session_mod, "_window_process", None)
        monkeypatch.setattr(session_mod, "SERVER_LOOP", None)

        handle = launcher.view(np.zeros((4, 4), dtype=np.float32), name="matlab-local")

        assert isinstance(handle, launcher.ViewHandle)
        assert native_calls
        assert native_calls[0]["url"].startswith("http://localhost:8123/shell?")
        assert browser_calls == []

    def test_jupyter_proxy_inline_uses_notebook_server_proxy(self, monkeypatch):
        pytest.importorskip("IPython.display")
        import arrayview._launcher as launcher

        monkeypatch.setattr(launcher, "_in_jupyter", lambda: True)
        monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: False)
        monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
        monkeypatch.setattr(launcher, "_server_pid", lambda port: os.getpid())
        monkeypatch.setattr(launcher, "_should_use_jupyter_proxy_inline", lambda: True)
        monkeypatch.setattr("arrayview._config.get_window_default", lambda _env: None)

        result = launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="nbclassic-inline",
            inline=True,
        )

        assert result.__class__.__name__ == "HTML"
        assert "proxy/8123/" in result.data
        assert "document.body && document.body.dataset" in result.data
        assert "frame.src = directSrc" in result.data
        assert "phase !== 'script-loaded'" in result.data

    def test_remote_vscode_jupyter_auto_opens_vscode_tab(self, monkeypatch):
        """VS Code tunnel notebook can't reach localhost through the webview sandbox,
        so `view(arr)` automatically routes to a VS Code webview tab instead of inline."""
        import arrayview._launcher as launcher

        calls = []

        monkeypatch.setattr(launcher, "_in_jupyter", lambda: True)
        monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: True)
        monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: True)
        monkeypatch.setattr(launcher, "_ensure_vscode_extension", lambda: True)
        monkeypatch.setattr(
            launcher,
            "_open_direct_via_shm",
            lambda data, name="array", title=None, floating=False: calls.append(
                {"shape": data.shape, "name": name, "title": title, "floating": floating}
            )
            or True,
        )

        result = launcher.view(np.zeros((4, 4), dtype=np.float32), name="remote-tab")

        assert result is None
        assert calls == [
            {
                "shape": (4, 4),
                "name": "remote-tab",
                "title": "ArrayView: remote-tab",
                "floating": False,
            }
        ]

    def test_jupyter_window_browser_disables_inline(self, monkeypatch):
        import arrayview._launcher as launcher

        opened = []

        monkeypatch.setattr(launcher, "_in_jupyter", lambda: True)
        monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: False)
        monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: False)
        monkeypatch.setattr(launcher, "_server_pid", lambda port: os.getpid())
        monkeypatch.setattr(
            launcher,
            "_open_browser",
            lambda url, **kwargs: opened.append({"url": url, **kwargs}),
        )
        monkeypatch.setattr("arrayview._config.get_window_default", lambda _env: None)

        handle = launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="browser-only",
            window="browser",
        )

        assert isinstance(handle, launcher.ViewHandle)
        assert opened
        assert opened[0]["url"].startswith("http://localhost:8123/?sid=")
