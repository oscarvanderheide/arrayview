"""Tests for --stack: lazy 4D/5D view over a directory of array files."""
import os

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from arrayview._io import _NiftiSeries, _load_nifti_series, load_data, load_data_with_meta


def _save_nifti(path, data, affine=None):
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return str(path)


def _make_patient_dir(root, name, n_files=1, shape=(4, 5, 6), dtype=np.float32, prefix="vol"):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_files):
        data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape) + i * 1000
        p = pdir / f"{prefix}_{i}.nii.gz"
        paths.append(_save_nifti(p, data))
    return paths


# ---------------------------------------------------------------------------
# 4D: one file per patient
# ---------------------------------------------------------------------------


class TestNiftiSeries4D:
    def test_shape_and_meta(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        _make_patient_dir(tmp_path, "p003")
        series, meta = load_data_with_meta(str(tmp_path))
        assert isinstance(series, _NiftiSeries)
        assert series.shape == (4, 5, 6, 3)
        assert series.ndim == 4
        assert meta is not None
        assert meta["axis_labels"] == ("R", "A", "S")

    def test_slicing_returns_correct_data(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        series, _ = load_data_with_meta(str(tmp_path))
        vol0 = np.asarray(nib.as_closest_canonical(nib.load(tmp_path / "p001" / "vol_0.nii.gz")).dataobj)
        vol1 = np.asarray(nib.as_closest_canonical(nib.load(tmp_path / "p002" / "vol_0.nii.gz")).dataobj)
        assert np.array_equal(series[:, :, :, 0], vol0)
        assert np.array_equal(series[:, :, :, 1], vol1)
        assert np.array_equal(series[:, :, 2, 0], vol0[:, :, 2])

    def test_lazy_only_caches_accessed_volumes(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        _make_patient_dir(tmp_path, "p003")
        series, _ = load_data_with_meta(str(tmp_path))
        _ = series[:, :, 0, 1]
        assert len(series._vol_cache) == 1
        assert (0, 1) in series._vol_cache or (1, 0) in series._vol_cache

    def test_patient_ordering_sorted(self, tmp_path):
        _make_patient_dir(tmp_path, "p010")
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p005")
        series, _ = load_data_with_meta(str(tmp_path))
        assert series.shape[-1] == 3
        vol_p001 = np.asarray(nib.as_closest_canonical(nib.load(tmp_path / "p001" / "vol_0.nii.gz")).dataobj)
        assert np.array_equal(series[..., 0], vol_p001)

    def test_load_data_dir(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        arr = load_data(str(tmp_path))
        assert isinstance(arr, _NiftiSeries)
        assert arr.shape == (4, 5, 6, 2)

    def test_non_nifti_files_ignored(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        (tmp_path / "p001" / "notes.txt").write_text("hello")
        (tmp_path / "p001" / "scan.dcm").write_bytes(b"\x00")
        series, _ = load_data_with_meta(str(tmp_path))
        assert series.shape == (4, 5, 6, 1)

    def test_nested_nifti_dirs_ignored_when_parent_has_series_files(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=2, prefix="mod")
        pdir = tmp_path / "p001"
        (pdir / "mod_0.nii.gz").rename(pdir / "T2_W.nii.gz")
        (pdir / "mod_1.nii.gz").rename(pdir / "MRCAT_W.nii.gz")
        masks_dir = pdir / "masks"
        masks_dir.mkdir()
        _save_nifti(masks_dir / "Brainstem.nii.gz", np.zeros((4, 5, 6), dtype=np.float32))

        series, _ = load_data_with_meta(str(tmp_path), select=["*T2_W*"])

        assert series.shape == (4, 5, 6, 1)


# ---------------------------------------------------------------------------
# 5D: multiple files per patient via --select
# ---------------------------------------------------------------------------


class TestNiftiSeries5D:
    def test_shape_with_select(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=3, prefix="mod")
        _make_patient_dir(tmp_path, "p002", n_files=3, prefix="mod")
        # Rename to modality-like names
        for pdir in ["p001", "p002"]:
            d = tmp_path / pdir
            for i, mod in enumerate(["t1", "t2", "flair"]):
                src = d / f"mod_{i}.nii.gz"
                dst = d / f"{mod}.nii.gz"
                src.rename(dst)
        series, meta = load_data_with_meta(str(tmp_path), select=["*t1*", "*t2*", "*flair*"])
        assert series.shape == (4, 5, 6, 2, 3)
        assert series.ndim == 5
        assert meta is not None

    def test_slicing_5d(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=2, prefix="m")
        _make_patient_dir(tmp_path, "p002", n_files=2, prefix="m")
        series, _ = load_data_with_meta(str(tmp_path), select=["*m_0*", "*m_1*"])
        vol0 = np.asarray(nib.as_closest_canonical(nib.load(tmp_path / "p001" / "m_0.nii.gz")).dataobj)
        vol1 = np.asarray(nib.as_closest_canonical(nib.load(tmp_path / "p001" / "m_1.nii.gz")).dataobj)
        assert np.array_equal(series[..., 0, 0], vol0)
        assert np.array_equal(series[..., 0, 1], vol1)

    def test_select_missing_pattern_error(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=1, prefix="t1")
        _make_patient_dir(tmp_path, "p002", n_files=1, prefix="t1")
        with pytest.raises(ValueError, match="no file matches"):
            load_data_with_meta(str(tmp_path), select=["*t1*", "*flair*"])

    def test_select_ambiguous_error(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=3, prefix="t1")
        with pytest.raises(ValueError, match="multiple files match"):
            load_data_with_meta(str(tmp_path), select=["*t1*"])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestNiftiSeriesErrors:
    def test_no_nifti_files(self, tmp_path):
        (tmp_path / "p001").mkdir()
        (tmp_path / "p001" / "scan.dcm").write_bytes(b"\x00")
        (tmp_path / "p001" / "notes.txt").write_text("hello")
        with pytest.raises(ValueError, match="No supported array files"):
            load_data_with_meta(str(tmp_path))

    def test_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No supported array files"):
            load_data_with_meta(str(tmp_path))

    def test_multiple_files_without_select(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=2)
        with pytest.raises(ValueError, match="--select"):
            load_data_with_meta(str(tmp_path))

    def test_shape_mismatch(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", shape=(4, 5, 6))
        _make_patient_dir(tmp_path, "p002", shape=(4, 5, 7))
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_data_with_meta(str(tmp_path))

    def test_dtype_mismatch(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", dtype=np.float32)
        _make_patient_dir(tmp_path, "p002", dtype=np.int16)
        with pytest.raises(ValueError, match="Dtype mismatch"):
            load_data_with_meta(str(tmp_path))


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


class TestViewDir:
    def test_view_dir_is_exported(self):
        import arrayview

        assert hasattr(arrayview, "view_dir")
        assert callable(arrayview.view_dir)

    def test_view_dir_is_lazy_marker(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        from arrayview._io import load_data_with_meta

        series, _ = load_data_with_meta(str(tmp_path))
        assert getattr(series, "_av_lazy", False) is True


# ---------------------------------------------------------------------------
# Non-NIfTI file series  (.npy, .npz, etc.)
# ---------------------------------------------------------------------------


class TestFileSeries:
    """Walk-and-stack with non-NIfTI array formats."""

    def test_npy_subdirs_4d(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for name in ("p001", "p002", "p003"):
            pdir = tmp_path / name
            pdir.mkdir()
            np.save(str(pdir / "vol.npy"), np.arange(120, dtype=np.float32).reshape(shape) + hash(name) % 100)

        series, meta = _io.load_data_with_meta(str(tmp_path))
        assert isinstance(series, _io._FileSeries)
        assert series.shape == (4, 5, 6, 3)
        assert meta is None

    def test_npy_5d_with_select(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for name in ("p001", "p002"):
            pdir = tmp_path / name
            pdir.mkdir()
            np.save(str(pdir / "t1.npy"), np.arange(120, dtype=np.float32).reshape(shape))
            np.save(str(pdir / "t2.npy"), np.arange(120, 240, dtype=np.float32).reshape(shape))

        series, _ = _io.load_data_with_meta(str(tmp_path), select=["*t1*", "*t2*"])
        assert isinstance(series, _io._FileSeries)
        assert series.shape == (4, 5, 6, 2, 2)

    def test_slicing_npy_series(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        vol0 = np.arange(120, dtype=np.float32).reshape(shape)
        vol1 = vol0 + 100
        (tmp_path / "p001").mkdir()
        (tmp_path / "p002").mkdir()
        np.save(str(tmp_path / "p001" / "vol.npy"), vol0)
        np.save(str(tmp_path / "p002" / "vol.npy"), vol1)

        series, _ = _io.load_data_with_meta(str(tmp_path))
        assert np.array_equal(series[:, :, :, 0], vol0)
        assert np.array_equal(series[:, :, :, 1], vol1)
        assert np.array_equal(series[:, :, 2, 0], vol0[:, :, 2])

    def test_shape_mismatch_error(self, tmp_path):
        import arrayview._io as _io

        (tmp_path / "p001").mkdir()
        (tmp_path / "p002").mkdir()
        np.save(str(tmp_path / "p001" / "vol.npy"), np.zeros((4, 5, 6)))
        np.save(str(tmp_path / "p002" / "vol.npy"), np.zeros((4, 5, 7)))

        with pytest.raises(ValueError, match="Shape mismatch"):
            _io.load_data_with_meta(str(tmp_path))
