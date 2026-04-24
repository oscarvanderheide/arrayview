"""Layer 1: HTTP API tests (no browser required, runs in seconds)."""

import io
import inspect
import os

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

    def test_shell_returns_html(self, client):
        r = client.get("/shell")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "document.createElement('iframe')" in r.text

    def test_launcher_cold_start_loading_infrastructure(self):
        import arrayview._launcher as launcher

        html = launcher._LOADING_HTML

        assert "#0c0c0c" in html  # dark background matches viewer theme
        assert "window.location.replace" in html  # JS navigates when server ready
        assert callable(launcher._run_loading_server)
        assert callable(launcher._with_loading)

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
        self, client, tmp_path
    ):
        arr = np.zeros((8, 6, 6, 6), dtype=np.float32)
        path = tmp_path / "memmap_4d.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        body = client.get(f"/metadata/{sid}").json()

        assert body["shape"] == [8, 6, 6, 6]
        assert body["default_dims"] == [2, 3]

    def test_4d_memmap_metadata_keeps_legacy_dims_when_trailing_plane_is_small(
        self, client, tmp_path
    ):
        arr = np.zeros((8, 8, 3, 3), dtype=np.float32)
        path = tmp_path / "memmap_small_trailing.npy"
        np.save(path, arr)

        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]
        body = client.get(f"/metadata/{sid}").json()

        assert body["shape"] == [8, 8, 3, 3]
        assert "default_dims" not in body

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
                        "overlay_sid": overlay_sid,
                    }
                )
                payload = ws.receive_bytes()

        rgba = np.frombuffer(payload[20:], dtype=np.uint8).reshape(6, 6, 4)
        assert rgba[2, 2, 3] > 0
        assert rgba[0, 0, 3] == 0


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
