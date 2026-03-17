"""Layer 1: HTTP API tests (no browser required, runs in seconds)."""

import io

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

    def test_shell_returns_html(self, client):
        r = client.get("/shell")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

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

    def test_unknown_sid_is_404(self, client):
        r = client.get("/metadata/doesnotexist000")
        assert r.status_code == 404

    def test_recommended_colormap_positive_data_is_gray(self, client, tmp_path):
        """Non-negative float data should recommend gray."""
        arr = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = tmp_path / "pos.npy"
        np.save(path, arr)
        r = client.post("/load", json={"filepath": str(path)})
        sid = r.json()["sid"]
        body = client.get(f"/metadata/{sid}").json()
        assert body["recommended_colormap"] == "gray"

    def test_recommended_colormap_signed_data_is_RdBu_r(self, client, tmp_path):
        """Signed float data (vmin < 0) should recommend RdBu_r."""
        arr = np.linspace(-1.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        path = tmp_path / "signed.npy"
        np.save(path, arr)
        r = client.post("/load", json={"filepath": str(path)})
        sid = r.json()["sid"]
        body = client.get(f"/metadata/{sid}").json()
        assert body["recommended_colormap"] == "RdBu_r"

    def test_recommended_colormap_bool_is_gray(self, client, tmp_path):
        """Boolean arrays should always recommend gray."""
        arr = np.array([[True, False], [False, True]])
        path = tmp_path / "bool.npy"
        np.save(path, arr)
        r = client.post("/load", json={"filepath": str(path)})
        sid = r.json()["sid"]
        body = client.get(f"/metadata/{sid}").json()
        assert body["recommended_colormap"] == "gray"


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


# ---------------------------------------------------------------------------
# /export_slice
# ---------------------------------------------------------------------------


class TestExportSlice:
    def test_returns_npy_file(self, client, sid_2d):
        r = client.get(
            f"/export_slice/{sid_2d}",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0", "complex_mode": 0},
        )
        assert r.status_code == 200
        assert "octet-stream" in r.headers["content-type"]
        arr = np.load(io.BytesIO(r.content))
        # arr_2d is 100×80; dim_x=1 (cols=80), dim_y=0 (rows=100)
        assert arr.shape == (100, 80)

    def test_filename_in_content_disposition(self, client, sid_2d):
        r = client.get(
            f"/export_slice/{sid_2d}",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0"},
        )
        assert "attachment" in r.headers.get("content-disposition", "")
        assert ".npy" in r.headers.get("content-disposition", "")

    def test_unknown_sid_is_404(self, client):
        r = client.get(
            "/export_slice/doesnotexist000",
            params={"dim_x": 1, "dim_y": 0, "indices": "0,0"},
        )
        assert r.status_code == 404


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
