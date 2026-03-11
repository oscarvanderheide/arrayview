"""
Phase 6: Large-array support tests.

Covers:
  - zarr_chunk_preset correctness
  - .zarr load parity with .npy (slice values identical)
  - guardrail behavior for FFT, GIF, grid on oversized arrays
  - /cache_info endpoint
  - adaptive cache budget (env-var override)
"""
import io
import os

import numpy as np
import pytest
import zarr
from PIL import Image

from arrayview._app import zarr_chunk_preset, HEAVY_OP_LIMIT_BYTES


# ---------------------------------------------------------------------------
# zarr_chunk_preset unit tests
# ---------------------------------------------------------------------------

class TestZarrChunkPreset:
    def test_1d_passthrough(self):
        assert zarr_chunk_preset((100,)) == (100,)

    def test_2d_full_slice(self):
        assert zarr_chunk_preset((256, 256)) == (256, 256)

    def test_2d_large_xy_tiles(self):
        # XY > ZARR_LARGE_XY_TILE → clamp to tile size
        c = zarr_chunk_preset((2048, 2048))
        assert c == (1024, 1024)

    def test_3d_one_z_per_chunk(self):
        assert zarr_chunk_preset((512, 512, 300)) == (512, 512, 1)

    def test_3d_large_xy_tiles(self):
        c = zarr_chunk_preset((2048, 2048, 100))
        assert c == (1024, 1024, 1)

    def test_4d_one_z_two_t(self):
        c = zarr_chunk_preset((256, 256, 50, 10))
        assert c == (256, 256, 1, 2)

    def test_4d_small_t_dim(self):
        # T=1 → clamp at 1
        c = zarr_chunk_preset((256, 256, 50, 1))
        assert c == (256, 256, 1, 1)

    def test_5d_full_channel(self):
        c = zarr_chunk_preset((256, 256, 50, 10, 4))
        assert c == (256, 256, 1, 1, 4)

    def test_6d_extra_dims_one(self):
        c = zarr_chunk_preset((128, 128, 10, 5, 3, 2))
        assert c[0] == 128 and c[1] == 128
        assert c[2:] == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# .zarr ↔ .npy parity tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npy_zarr_pair(tmp_path_factory):
    """Create a small 3D .npy + .zarr pair with identical data.  Module-scoped
    so we only create them once for all parity tests."""
    tmp = tmp_path_factory.mktemp("parity")
    arr = np.random.default_rng(99).standard_normal((20, 32, 32)).astype(np.float32)

    npy_path = tmp / "arr.npy"
    zarr_path = tmp / "arr.zarr"

    np.save(npy_path, arr)

    chunks = zarr_chunk_preset(arr.shape)
    z = zarr.open(str(zarr_path), mode="w", shape=arr.shape, dtype=arr.dtype, chunks=chunks)
    z[:] = arr

    return str(npy_path), str(zarr_path), arr


class TestZarrNpyParity:
    """Slice values from zarr and npy sessions must be numerically identical."""

    def _load_sid(self, client, path):
        r = client.post("/load", json={"filepath": path})
        r.raise_for_status()
        return r.json()["sid"]

    def test_slice_value_parity_axis0(self, client, npy_zarr_pair):
        npy_path, zarr_path, arr = npy_zarr_pair
        npy_sid  = self._load_sid(client, npy_path)
        zarr_sid = self._load_sid(client, zarr_path)

        for idx in [0, 10, 19]:
            params = dict(dim_x=2, dim_y=1, indices=f"{idx},0,0",
                          colormap="gray", dr=0)
            rn = client.get(f"/slice/{npy_sid}",  params=params)
            rz = client.get(f"/slice/{zarr_sid}", params=params)
            assert rn.status_code == 200 and rz.status_code == 200

            img_n = np.array(Image.open(io.BytesIO(rn.content)).convert("RGB"))
            img_z = np.array(Image.open(io.BytesIO(rz.content)).convert("RGB"))
            # Pixel values should be identical (same render path, same LUT)
            np.testing.assert_array_equal(img_n, img_z,
                err_msg=f"Slice {idx} pixel mismatch between .npy and .zarr")

    def test_metadata_parity(self, client, npy_zarr_pair):
        npy_path, zarr_path, arr = npy_zarr_pair
        npy_sid  = self._load_sid(client, npy_path)
        zarr_sid = self._load_sid(client, zarr_path)

        info_n = client.get(f"/info/{npy_sid}").json()
        info_z = client.get(f"/info/{zarr_sid}").json()

        assert info_n["shape"] == info_z["shape"]
        assert info_n["ndim"]  == info_z["ndim"]
        # dtype may differ slightly in string repr; check shape/ndim only


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------

def _register_synthetic(client, shape, tmp_path):
    """Create a tiny .npy with the given shape and register it."""
    arr = np.zeros(shape, dtype=np.float32)
    path = tmp_path / "big.npy"
    np.save(path, arr)
    r = client.post("/load", json={"filepath": str(path)})
    r.raise_for_status()
    return r.json()["sid"]


class TestGuardrails:
    """Heavy operations must be blocked when array size exceeds HEAVY_OP_LIMIT_BYTES."""

    @pytest.fixture
    def oversized_sid(self, client, tmp_path):
        """Register an array whose byte size is guaranteed > HEAVY_OP_LIMIT_BYTES.

        We use a shape whose total > limit but don't actually allocate that much
        RAM — we trick the server by registering a tiny on-disk .npy but then
        monkeypatching SESSIONS to hold a large shape.  Instead, use a shape
        that's truly large but zero-filled so numpy allocates it lazily.

        Simpler: use a shape where n_slices * slice_bytes > limit, tested via
        the GIF/grid endpoints which compute the estimate from session.shape.
        """
        # 512*512*300*4 = 314 MB — under the 500 MB limit.
        # To exceed it: 1024*1024*130*4 = 549 MB > 500 MB.
        # BUT we don't want to allocate 549 MB in the test process.
        # Hack: save a (1, 1, 1) array, then monkey-patch the session shape.
        arr = np.zeros((1, 1, 1), dtype=np.float32)
        path = tmp_path / "fake_big.npy"
        np.save(path, arr)
        r = client.post("/load", json={"filepath": str(path)})
        r.raise_for_status()
        sid = r.json()["sid"]

        # Patch the session shape so the guardrail sees a large array
        from arrayview._app import SESSIONS
        session = SESSIONS[sid]
        session.shape = (1024, 1024, 130)  # 549 MB > 500 MB
        return sid

    def test_fft_blocked_on_large_array(self, client, oversized_sid):
        r = client.post(f"/fft/{oversized_sid}",
                        json={"axes": "0"},
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert "error" in body
        assert body.get("too_large") is True
        assert "FFT blocked" in body["error"]

    def test_grid_blocked_on_large_array(self, client, oversized_sid):
        r = client.get(f"/grid/{oversized_sid}", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0,0",
            "slice_dim": 2, "colormap": "gray", "dr": 0,
        })
        assert r.status_code == 400
        body = r.json()
        assert body.get("too_large") is True
        assert "Grid blocked" in body["error"]

    def test_gif_blocked_on_large_array(self, client, oversized_sid):
        r = client.get(f"/gif/{oversized_sid}", params={
            "dim_x": 1, "dim_y": 0, "indices": "0,0,0",
            "slice_dim": 2, "colormap": "gray", "dr": 0,
        })
        assert r.status_code == 400
        body = r.json()
        assert body.get("too_large") is True
        assert "GIF blocked" in body["error"]

    def test_small_array_not_blocked_by_fft(self, client, sid_3d):
        """3D 20×64×64 = 10 MB — well under limit, FFT should proceed."""
        r = client.post(f"/fft/{sid_3d}",
                        json={"axes": "2"},
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert "error" not in body or body.get("too_large") is not True
        # Restore: toggle FFT off
        client.post(f"/fft/{sid_3d}",
                    json={"axes": ""},
                    headers={"Content-Type": "application/json"})


# ---------------------------------------------------------------------------
# /cache_info endpoint
# ---------------------------------------------------------------------------

class TestCacheInfo:
    def test_cache_info_returns_structure(self, client, sid_3d):
        r = client.get(f"/cache_info/{sid_3d}")
        assert r.status_code == 200
        body = r.json()
        for key in ("raw_cache", "rgba_cache", "mosaic_cache"):
            assert key in body
            assert "entries" in body[key]
            assert "used_bytes" in body[key]
            assert "budget_bytes" in body[key]
            assert "used_mb" in body[key]
            assert "budget_mb" in body[key]
        assert "heavy_op_limit_mb" in body

    def test_cache_info_budget_positive(self, client, sid_3d):
        body = client.get(f"/cache_info/{sid_3d}").json()
        assert body["raw_cache"]["budget_bytes"] > 0
        assert body["rgba_cache"]["budget_bytes"] > 0
        assert body["mosaic_cache"]["budget_bytes"] > 0

    def test_cache_info_unknown_sid_is_404(self, client):
        r = client.get("/cache_info/doesnotexist000")
        assert r.status_code == 404

    def test_cache_info_entries_grow_after_slice(self, client, sid_3d):
        # Warm the cache with one slice
        client.get(f"/slice/{sid_3d}", params={
            "dim_x": 2, "dim_y": 1, "indices": "0,0,0",
            "colormap": "gray", "dr": 0,
        })
        body = client.get(f"/cache_info/{sid_3d}").json()
        # At least one slice should be in the raw or rgba cache
        total_entries = (
            body["raw_cache"]["entries"] +
            body["rgba_cache"]["entries"]
        )
        assert total_entries >= 1


# ---------------------------------------------------------------------------
# Adaptive cache env-var override
# ---------------------------------------------------------------------------

class TestAdaptiveCacheBudget:
    def test_env_var_overrides_raw_cache(self, monkeypatch):
        monkeypatch.setenv("ARRAYVIEW_RAW_CACHE_MB", "42")
        # Re-evaluate the helper directly (not module-level constant)
        from arrayview._app import _cache_budget
        result = _cache_budget("ARRAYVIEW_RAW_CACHE_MB", 0.05)
        assert result == 42 * 1024 * 1024

    def test_fallback_uses_fraction(self, monkeypatch):
        monkeypatch.delenv("ARRAYVIEW_RAW_CACHE_MB", raising=False)
        from arrayview._app import _cache_budget, _total_ram_bytes
        ram = _total_ram_bytes()
        expected = max(64 * 1024 * 1024, int(ram * 0.05))
        result = _cache_budget("ARRAYVIEW_RAW_CACHE_MB", 0.05)
        assert result == expected

    def test_bad_env_var_falls_back_to_fraction(self, monkeypatch):
        monkeypatch.setenv("ARRAYVIEW_RAW_CACHE_MB", "notanumber")
        from arrayview._app import _cache_budget, _total_ram_bytes
        ram = _total_ram_bytes()
        expected = max(64 * 1024 * 1024, int(ram * 0.05))
        result = _cache_budget("ARRAYVIEW_RAW_CACHE_MB", 0.05)
        assert result == expected
