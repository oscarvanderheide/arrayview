"""Tests for NIfTI canonical reorient + spatial metadata extraction."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from arrayview._io import _load_nifti_with_meta, load_data_with_meta


def _save_nifti(tmp_path, affine, shape=(8, 9, 10)):
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    img = nib.Nifti1Image(data, affine)
    path = tmp_path / "test.nii.gz"
    nib.save(img, str(path))
    return str(path)


def test_axis_aligned_nifti(tmp_path):
    # Standard RAS affine, 1 mm isotropic
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-10, -20, -30]
    path = _save_nifti(tmp_path, affine)
    arr, meta = load_data_with_meta(path)
    assert meta is not None
    assert arr.ndim == 3
    assert meta["axis_labels"] == ("R", "A", "S")
    assert meta["voxel_sizes"] == pytest.approx((1.0, 1.0, 1.0))
    assert meta["is_oblique"] is False


def test_anisotropic_axis_aligned(tmp_path):
    affine = np.diag([2.0, 1.5, 0.8, 1.0])
    path = _save_nifti(tmp_path, affine)
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
    path = _save_nifti(tmp_path, rot)
    _, meta = load_data_with_meta(path)
    assert meta["is_oblique"] is True
    assert meta["voxel_sizes"] == pytest.approx((1.0, 1.0, 1.0))


def test_non_nifti_returns_none(tmp_path):
    p = tmp_path / "x.npy"
    np.save(p, np.zeros((3, 3)))
    arr, meta = load_data_with_meta(str(p))
    assert meta is None
    assert arr.shape == (3, 3)
