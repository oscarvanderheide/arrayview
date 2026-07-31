"""Tests for native-order NIfTI loading and spatial metadata extraction."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from arrayview._io import _load_nifti_with_meta, load_data_with_meta
from arrayview._session import _recommend_colormap_reason


def _save_nifti(tmp_path, affine, shape=(8, 9, 10), filename="test.nii.gz"):
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / filename
    nib.save(img, str(path))
    return str(path), data


def test_axis_aligned_nifti(tmp_path):
    # Standard RAS affine, 1 mm isotropic
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-10, -20, -30]
    path, _ = _save_nifti(tmp_path, affine)
    arr, meta = load_data_with_meta(path)
    assert meta is not None
    assert arr.ndim == 3
    assert meta["axis_labels"] == ("R", "A", "S")
    assert meta["voxel_sizes"] == pytest.approx((1.0, 1.0, 1.0))
    assert meta["is_oblique"] is False


def test_anisotropic_axis_aligned(tmp_path):
    affine = np.diag([2.0, 1.5, 0.8, 1.0])
    path, _ = _save_nifti(tmp_path, affine)
    _, meta = load_data_with_meta(path)
    assert meta["voxel_sizes"] == pytest.approx((2.0, 1.5, 0.8))
    assert meta["is_oblique"] is False


def test_oblique_nifti(tmp_path):
    # 15-degree rotation about Z applied to identity
    theta = np.deg2rad(15.0)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    path, _ = _save_nifti(tmp_path, rot)
    _, meta = load_data_with_meta(path)
    assert meta["is_oblique"] is True
    assert meta["voxel_sizes"] == pytest.approx((1.0, 1.0, 1.0))


def test_non_nifti_returns_none(tmp_path):
    p = tmp_path / "x.npy"
    np.save(p, np.zeros((3, 3)))
    arr, meta = load_data_with_meta(str(p))
    assert meta is None
    assert arr.shape == (3, 3)


def test_uncompressed_nifti_stays_proxy_backed_in_native_order(tmp_path, monkeypatch):
    """Loading a .nii must not reorient or materialize its voxel payload."""
    affine = np.array(
        [
            [0.0, -3.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path, expected = _save_nifti(
        tmp_path, affine, shape=(3, 4, 5), filename="native-order.nii"
    )

    monkeypatch.setattr(
        nib,
        "as_closest_canonical",
        lambda _img: pytest.fail("uncompressed NIfTI must not be canonicalized"),
    )
    monkeypatch.setattr(
        nib.arrayproxy.ArrayProxy,
        "__array__",
        lambda self, dtype=None: pytest.fail(
            "uncompressed NIfTI must not be fully materialized"
        ),
    )

    arr, _ = _load_nifti_with_meta(path)

    assert isinstance(arr, nib.arrayproxy.ArrayProxy)
    assert arr.shape == expected.shape
    np.testing.assert_array_equal(arr[:, :, 2], expected[:, :, 2])


def test_permuted_flipped_affine_reports_source_orientation(tmp_path):
    """Spatial labels and voxel sizes describe native axes, not a hidden RAS view."""
    affine = np.array(
        [
            [0.0, -3.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path, _ = _save_nifti(
        tmp_path, affine, shape=(3, 4, 5), filename="source-orientation.nii"
    )

    arr, meta = _load_nifti_with_meta(path)

    assert arr.shape == (3, 4, 5)
    assert meta["axis_labels"] == ("A", "L", "S")
    assert meta["voxel_sizes"] == pytest.approx((2.0, 3.0, 4.0))
    np.testing.assert_array_equal(meta["affine"], affine)


def test_colormap_recommendation_uses_bounded_indexing_without_flattening():
    class GuardedProxy:
        dtype = np.dtype(np.float32)
        shape = (1_000_000, 1_000, 100)

        def __init__(self):
            self.keys = []

        def __array__(self, dtype=None):
            raise AssertionError("colormap sampling converted the full proxy")

        def ravel(self, *args, **kwargs):
            raise AssertionError("colormap sampling flattened the full proxy")

        def __getitem__(self, key):
            self.keys.append(key)
            return np.array([-1.0, 2.0], dtype=np.float32)

    data = GuardedProxy()

    reason = _recommend_colormap_reason(data)

    assert reason == "RdBu_r (signed data — vmin < 0)"
    assert data.keys, "colormap recommendation should inspect a bounded sample"


def test_explicit_ras_conversion_reorients_native_volume(client, tmp_path):
    affine = np.array(
        [
            [0.0, -3.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path, _ = _save_nifti(
        tmp_path, affine, shape=(3, 4, 5), filename="explicit-ras.nii"
    )
    loaded = client.post("/load", json={"filepath": path})
    loaded.raise_for_status()
    sid = loaded.json()["sid"]

    native = client.get(f"/metadata/{sid}").json()
    assert native["shape"] == [3, 4, 5]
    assert native["spatial_meta"]["axis_labels"] == ["A", "L", "S"]
    assert native["spatial_meta"]["is_canonical"] is False

    converted = client.post(f"/resample_ras/{sid}", json={"enabled": True})
    converted.raise_for_status()
    assert converted.json()["shape"] == [4, 3, 5]
    active = client.get(f"/info/{sid}").json()
    assert active["ras_resample_active"] is True

    reverted = client.post(f"/resample_ras/{sid}", json={"enabled": False})
    reverted.raise_for_status()
    assert reverted.json()["shape"] == [3, 4, 5]
