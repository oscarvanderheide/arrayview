"""Tests for directory FILE stacking: lazy 4D/5D view over array files."""
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from arrayview._io import _NiftiSeries, _RaggedFileSeries, _load_nifti_series, load_data, load_data_with_meta


def _wait_for_volume(series, key=(0, 0), timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with series._cache_lock:
            if key in series._vol_cache:
                return series._vol_cache[key]
        time.sleep(0.005)
    raise AssertionError(f"volume {key} was not cached within {timeout} seconds")


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
        _wait_for_volume(series, (1, 0))
        assert len(series._vol_cache) == 1
        assert (1, 0) in series._vol_cache

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
        _make_patient_dir(tmp_path, "p001", n_files=1, prefix="mod")
        pdir = tmp_path / "p001"
        (pdir / "mod_0.nii.gz").rename(pdir / "T2_W.nii.gz")
        masks_dir = pdir / "masks"
        masks_dir.mkdir()
        _save_nifti(masks_dir / "Brainstem.nii.gz", np.zeros((4, 5, 6), dtype=np.float32))

        series, _ = load_data_with_meta(str(tmp_path))

        assert series.shape == (4, 5, 6, 1)


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

    def test_multiple_files_recommend_stack_patterns(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", n_files=2)
        with pytest.raises(ValueError, match="--stack"):
            load_data_with_meta(str(tmp_path))

    def test_shape_mismatch_falls_back_to_ragged(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", shape=(4, 5, 6))
        _make_patient_dir(tmp_path, "p002", shape=(4, 5, 7))
        series, _ = load_data_with_meta(str(tmp_path))
        assert isinstance(series, _RaggedFileSeries)
        assert series.ragged_spatial_shapes == [[(4, 5, 6)], [(4, 5, 7)]]

    def test_dense_shape_mismatch_is_explicit_error(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", shape=(4, 5, 6))
        _make_patient_dir(tmp_path, "p002", shape=(4, 5, 7))
        with pytest.raises(ValueError, match="Dense stacking"):
            load_data_with_meta(str(tmp_path), stack="dense")

    def test_header_scan_does_not_cache_voxels(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        series, _ = load_data_with_meta(str(tmp_path))
        assert series._vol_cache == {}

    def test_header_scan_does_not_canonicalize_voxels(self, tmp_path, monkeypatch):
        _make_patient_dir(tmp_path, "p001")
        monkeypatch.setattr(
            nib,
            "as_closest_canonical",
            lambda _img: (_ for _ in ()).throw(
                AssertionError("header scan materialized voxel orientation")
            ),
        )

        series, _ = load_data_with_meta(str(tmp_path))

        assert series._vol_cache == {}

    def test_eager_load_populates_cache(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        series, _ = load_data_with_meta(str(tmp_path), load="eager")
        assert len(series._vol_cache) == 2

    def test_concurrent_access_decompresses_volume_once(self, tmp_path, monkeypatch):
        _make_patient_dir(tmp_path, "p001")
        series, _ = load_data_with_meta(str(tmp_path))
        original_load = nib.load
        loads = []

        def counted_load(path, *args, **kwargs):
            loads.append(str(path))
            time.sleep(0.05)
            return original_load(path, *args, **kwargs)

        monkeypatch.setattr(nib, "load", counted_load)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(series[:, :, 2, 0]))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(loads) == 1
        assert len(results) == 2

    def test_compressed_float64_uses_float32_display_cache(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", dtype=np.float64)
        series, _ = load_data_with_meta(str(tmp_path))

        displayed = series[:, :, 2, 0]
        cached = _wait_for_volume(series)

        assert displayed.dtype == np.float32
        assert cached.dtype == np.float32

    def test_compressed_axial_plane_returns_before_full_volume(self, tmp_path, monkeypatch):
        from arrayview import _io

        _make_patient_dir(tmp_path, "p001", shape=(16, 17, 8))
        series, _ = load_data_with_meta(str(tmp_path))
        original_read = _io._read_exact
        second_plane_started = threading.Event()
        release_background = threading.Event()
        reads = 0

        def controlled_read(stream, size):
            nonlocal reads
            reads += 1
            if reads == 2:
                second_plane_started.set()
                assert release_background.wait(2.0)
            return original_read(stream, size)

        monkeypatch.setattr(_io, "_read_exact", controlled_read)
        try:
            plane = series[:, :, 0, 0]
            assert second_plane_started.wait(1.0)
            assert (0, 0) not in series._vol_cache
            expected = np.asarray(
                nib.as_closest_canonical(
                    nib.load(tmp_path / "p001" / "vol_0.nii.gz")
                ).dataobj
            )
            assert np.array_equal(plane, expected[:, :, 0])
        finally:
            release_background.set()
        _wait_for_volume(series)
        assert reads == 8

    def test_progressive_plane_preserves_orientation_scaling_and_cache(self, tmp_path):
        pdir = tmp_path / "p001"
        pdir.mkdir()
        path = pdir / "vol_0.nii.gz"
        data = np.arange(5 * 6 * 7, dtype=np.float64).reshape(5, 6, 7)
        affine = np.array(
            [[0.0, -2.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0],
             [0.0, 0.0, -3.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        image = nib.Nifti1Image(data, affine)
        image.header.set_slope_inter(2.5, -3.0)
        nib.save(image, path)
        series, _ = load_data_with_meta(str(tmp_path))
        expected = np.asarray(
            nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32
        )

        plane = series[:, :, 2, 0]
        cached = _wait_for_volume(series)

        assert np.array_equal(plane, expected[:, :, 2])
        assert np.array_equal(cached, expected)

    def test_non_axial_source_layout_falls_back_to_full_load(self, tmp_path, monkeypatch):
        from arrayview import _io

        pdir = tmp_path / "p001"
        pdir.mkdir()
        path = pdir / "vol_0.nii.gz"
        data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        affine = np.array(
            [[0.0, 0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0],
             [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        nib.save(nib.Nifti1Image(data, affine), path)
        series, _ = load_data_with_meta(str(tmp_path))
        monkeypatch.setattr(
            _io,
            "_read_exact",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("unsupported layout used progressive decoder")
            ),
        )

        plane = series[:, :, 1, 0]
        expected = np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj)

        assert np.array_equal(plane, expected[:, :, 1])
        assert (0, 0) in series._vol_cache

    def test_neighbor_decode_does_not_cancel_visible_patient(self, tmp_path, monkeypatch):
        from arrayview import _io

        _make_patient_dir(tmp_path, "p001", shape=(12, 13, 6))
        _make_patient_dir(tmp_path, "p002", shape=(12, 13, 6))
        series, _ = load_data_with_meta(str(tmp_path))
        original_read = _io._read_exact
        first_path = str(tmp_path / "p001" / "vol_0.nii.gz")
        old_background_started = threading.Event()
        release_old = threading.Event()
        reads_by_path = {}

        def controlled_read(stream, size):
            path = str(stream.name)
            reads_by_path[path] = reads_by_path.get(path, 0) + 1
            if path == first_path and reads_by_path[path] == 2:
                old_background_started.set()
                assert release_old.wait(2.0)
            return original_read(stream, size)

        monkeypatch.setattr(_io, "_read_exact", controlled_read)
        try:
            _ = series[:, :, 0, 0]
            assert old_background_started.wait(1.0)
            _ = series[:, :, 0, 1]
            with series._progressive_lock:
                assert len(series._progressive_jobs) <= 2
                assert (0, 0) in series._progressive_jobs
        finally:
            release_old.set()
        _wait_for_volume(series, (0, 0))
        _wait_for_volume(series, (1, 0))

    def test_overlay_prefetch_jobs_cannot_queue_base_patient(self, tmp_path, monkeypatch):
        from concurrent.futures import Future
        from arrayview import _io

        overlay_paths = [
            _make_patient_dir(tmp_path, f"overlay{i}", shape=(10, 11, 4))[0]
            for i in range(2)
        ]
        base_path = _make_patient_dir(tmp_path, "base", shape=(10, 11, 4))[0]
        overlays = [
            _NiftiSeries([[path]], (10, 11, 4), np.float32)
            for path in overlay_paths
        ]
        base = _NiftiSeries([[base_path]], (10, 11, 4), np.float32)
        queued = []
        two_overlays_queued = threading.Event()
        selected_lanes = []

        class QueuedPool:
            def submit(self, fn):
                future = Future()
                queued.append((fn, future))
                if len(queued) == 2:
                    two_overlays_queued.set()
                return future

        class ImmediatePool:
            def submit(self, fn):
                future = Future()
                assert future.set_running_or_notify_cancel()
                try:
                    fn()
                except Exception as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(None)
                return future

        overlay_pool = QueuedPool()
        base_pool = ImmediatePool()

        def choose_pool(*, overlay_prefetch=False):
            selected_lanes.append(overlay_prefetch)
            return overlay_pool if overlay_prefetch else base_pool

        monkeypatch.setattr(_io, "_get_progressive_nifti_pool", choose_pool)
        overlay_results = []
        threads = [
            threading.Thread(
                target=lambda series=series: overlay_results.append(
                    series[:, :, 0, 0]
                ),
                name=f"arrayview-overlay-prefetch_{i}",
            )
            for i, series in enumerate(overlays)
        ]
        for thread in threads:
            thread.start()
        assert two_overlays_queued.wait(1.0)
        assert all(series._progressive_jobs[(0, 0)]._volume is None for series in overlays)

        base_plane = base[:, :, 0, 0]

        expected = np.asarray(
            nib.as_closest_canonical(nib.load(base_path)).dataobj
        )
        assert np.array_equal(base_plane, expected[:, :, 0])
        assert selected_lanes == [True, True, False]

        for fn, future in queued:
            assert future.set_running_or_notify_cancel()
            fn()
            future.set_result(None)
        for thread in threads:
            thread.join(1.0)
            assert not thread.is_alive()
        assert len(overlay_results) == 2

    def test_ragged_nifti_stack_uses_progressive_axial_load(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", shape=(8, 9, 5))
        _make_patient_dir(tmp_path, "p002", shape=(8, 9, 7))
        series, _ = load_data_with_meta(str(tmp_path))
        assert isinstance(series, _RaggedFileSeries)
        expected = np.asarray(
            nib.as_closest_canonical(
                nib.load(tmp_path / "p002" / "vol_0.nii.gz")
            ).dataobj
        )

        plane = series[:, :, 3, 1]
        cached = _wait_for_volume(series, (1, 0))

        assert np.array_equal(plane, expected[:, :, 3])
        assert np.array_equal(cached, expected)

    def test_dtype_mismatch(self, tmp_path):
        _make_patient_dir(tmp_path, "p001", dtype=np.float32)
        _make_patient_dir(tmp_path, "p002", dtype=np.int16)
        with pytest.raises(ValueError, match="Dtype mismatch"):
            load_data_with_meta(str(tmp_path))


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


class TestPythonApi:
    def test_disk_collection_helpers_are_not_exported(self):
        import arrayview

        assert not hasattr(arrayview, "view_dir")
        assert not hasattr(arrayview, "view_dir_patterns")

    def test_directory_series_is_lazy_marker(self, tmp_path):
        _make_patient_dir(tmp_path, "p001")
        _make_patient_dir(tmp_path, "p002")
        from arrayview._io import load_data_with_meta

        series, _ = load_data_with_meta(str(tmp_path))
        assert getattr(series, "_av_lazy", False) is True


def test_collection_prefetch_only_warms_one_neighbor(monkeypatch):
    from arrayview import _session as session_mod
    import arrayview._render as render_mod

    warmed = []
    session = SimpleNamespace(
        shape=(32, 32, 20, 100),
        data=SimpleNamespace(_stack_axes=(3,)),
        raw_cache={},
    )

    class ImmediatePool:
        def submit(self, fn):
            fn()

    monkeypatch.setattr(session_mod, "_get_prefetch_pool", lambda: ImmediatePool())
    monkeypatch.setattr(
        render_mod,
        "extract_slice",
        lambda _session, _dx, _dy, idx: warmed.append(idx[3]),
    )

    session_mod._schedule_prefetch(session, 0, 1, [0, 0, 10, 12], 3, 1)

    assert warmed == [13]


def test_stale_collection_prefetch_is_coalesced(monkeypatch):
    from arrayview import _session as session_mod
    import arrayview._render as render_mod

    queued = []
    warmed = []
    session = SimpleNamespace(
        shape=(32, 32, 20, 100),
        data=SimpleNamespace(_stack_axes=(3,)),
        raw_cache={},
    )

    class QueuedPool:
        def submit(self, fn):
            queued.append(fn)

    monkeypatch.setattr(session_mod, "_get_prefetch_pool", lambda: QueuedPool())
    monkeypatch.setattr(
        render_mod,
        "extract_slice",
        lambda _session, _dx, _dy, idx: warmed.append(idx[3]),
    )

    session_mod._schedule_prefetch(session, 0, 1, [0, 0, 10, 12], 3, 1)
    session_mod._schedule_prefetch(session, 0, 1, [0, 0, 10, 20], 3, 1)
    for job in queued:
        job()

    assert warmed == [21]


def test_overlay_prefetch_only_warms_visible_next_patient(monkeypatch):
    from arrayview import _session as session_mod
    import arrayview._render as render_mod

    warmed = []
    main = SimpleNamespace(
        shape=(32, 32, 20, 100),
        _overlay_prefetch_generation=0,
    )
    overlays = {
        "visible": SimpleNamespace(
            shape=main.shape,
            data=SimpleNamespace(_stack_axes=(3,)),
            raw_cache={},
        ),
        "hidden": SimpleNamespace(
            shape=main.shape,
            data=SimpleNamespace(_stack_axes=(3,)),
            raw_cache={},
        ),
        "missing": SimpleNamespace(
            shape=main.shape,
            data=SimpleNamespace(
                _stack_axes=(3,),
                _file_matrix=[["mask.nii.gz"] for _ in range(100)],
            ),
            raw_cache={},
        ),
        "not_collection": SimpleNamespace(
            shape=main.shape,
            data=SimpleNamespace(_stack_axes=()),
            raw_cache={},
        ),
    }
    overlays["missing"].data._file_matrix[13][0] = None

    class ImmediatePool:
        def submit(self, fn, *args):
            fn(*args)

    monkeypatch.setattr(session_mod, "SESSIONS", overlays)
    monkeypatch.setattr(session_mod, "_get_overlay_prefetch_pool", lambda: ImmediatePool())
    monkeypatch.setattr(
        render_mod,
        "extract_slice",
        lambda overlay, _dx, _dy, idx: warmed.append((overlay, idx[3])),
    )

    session_mod._schedule_overlay_prefetch(
        main,
        "visible,hidden,missing,not_collection,unknown",
        "0.45,0,0.45,0.45,0.45",
        0,
        1,
        [0, 0, 10, 12],
        3,
        1,
    )

    assert warmed == [(overlays["visible"], 13)]


def test_stale_overlay_prefetch_is_coalesced(monkeypatch):
    from arrayview import _session as session_mod
    import arrayview._render as render_mod

    queued = []
    warmed = []
    main = SimpleNamespace(shape=(32, 32, 20, 100))
    overlay = SimpleNamespace(
        shape=main.shape,
        data=SimpleNamespace(_stack_axes=(3,)),
        raw_cache={},
    )

    class QueuedPool:
        def submit(self, fn, *args):
            queued.append((fn, args))

    monkeypatch.setattr(session_mod, "SESSIONS", {"overlay": overlay})
    monkeypatch.setattr(session_mod, "_get_overlay_prefetch_pool", lambda: QueuedPool())
    monkeypatch.setattr(
        render_mod,
        "extract_slice",
        lambda _overlay, _dx, _dy, idx: warmed.append(idx[3]),
    )

    session_mod._schedule_overlay_prefetch(
        main, "overlay", "0.45", 0, 1, [0, 0, 10, 12], 3, 1
    )
    session_mod._schedule_overlay_prefetch(
        main, "overlay", "0.45", 0, 1, [0, 0, 11, 12], 3, 1
    )
    session_mod._schedule_overlay_prefetch(
        main, "overlay", "0.45", 0, 1, [0, 0, 10, 20], 3, 1
    )
    for fn, args in queued:
        fn(*args)

    assert warmed == [21]


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

    def test_shape_mismatch_uses_ragged_collection(self, tmp_path):
        import arrayview._io as _io

        (tmp_path / "p001").mkdir()
        (tmp_path / "p002").mkdir()
        np.save(str(tmp_path / "p001" / "vol.npy"), np.zeros((4, 5, 6)))
        np.save(str(tmp_path / "p002" / "vol.npy"), np.zeros((4, 5, 7)))

        series, _ = _io.load_data_with_meta(str(tmp_path))
        assert isinstance(series, _io._RaggedFileSeries)


class TestDirCollection:
    def test_image_patterns_stack_channels_and_overlay_roles(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case_idx, case in enumerate(("caseA", "caseB")):
            for sub in ("images", "gt", "pred"):
                (tmp_path / sub).mkdir(exist_ok=True)
            np.save(
                tmp_path / "images" / f"{case}_0000.npy",
                np.full(shape, case_idx, dtype=np.float32),
            )
            np.save(
                tmp_path / "images" / f"{case}_0001.npy",
                np.full(shape, case_idx + 10, dtype=np.float32),
            )
            np.save(
                tmp_path / "gt" / f"{case}.npy",
                np.full(shape, case_idx + 1, dtype=np.uint8),
            )
            np.save(
                tmp_path / "pred" / f"{case}.npy",
                np.full(shape, case_idx + 2, dtype=np.uint8),
            )

        data, meta, overlays, summary = _io.load_dir_collection(
            [
                str(tmp_path / "images" / "*_0000.npy"),
                str(tmp_path / "images" / "*_0001.npy"),
            ],
            overlays=[
                ("gt", str(tmp_path / "gt" / "*.npy")),
                ("pred", str(tmp_path / "pred" / "*.npy")),
            ],
        )

        assert isinstance(data, _io._FileSeries)
        assert meta is None
        assert data.shape == (4, 5, 6, 2, 2)
        assert overlays[0]["name"] == "gt"
        assert overlays[0]["data"].shape == (4, 5, 6, 2)
        assert overlays[1]["name"] == "pred"
        assert summary["cases"] == ["caseA", "caseB"]
        assert np.array_equal(data[:, :, :, 1, 1], np.full(shape, 11, dtype=np.float32))
        assert np.array_equal(overlays[0]["data"][:, :, :, 1], np.full(shape, 2, dtype=np.uint8))

    def test_overlay_missing_case_errors(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case in ("caseA", "caseB"):
            (tmp_path / "images").mkdir(exist_ok=True)
            np.save(tmp_path / "images" / f"{case}_0000.npy", np.zeros(shape))
        (tmp_path / "gt").mkdir()
        np.save(tmp_path / "gt" / "caseA.npy", np.zeros(shape, dtype=np.uint8))

        with pytest.raises(ValueError, match="missing case"):
            _io.load_dir_collection(
                [str(tmp_path / "images" / "*_0000.npy")],
                overlays=[("gt", str(tmp_path / "gt" / "*.npy"))],
                case_regex=r"(?P<case>case[A-Z])",
            )

    def test_sparse_overlay_missing_case_is_empty(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        for case in ("caseA", "caseB"):
            np.save(images / f"{case}.npy", np.ones(shape, dtype=np.float32))
        np.save(masks / "caseA.npy", np.ones(shape, dtype=np.uint8))

        _data, _meta, overlays, summary = _io.load_dir_collection(
            [str(images / "*.npy")],
            overlays=[("organ", str(masks / "*.npy"), True)],
            case_regex=r"(?P<case>case[A-Z])",
        )

        overlay = overlays[0]["data"]
        assert np.array_equal(overlay[..., 0], np.ones(shape, dtype=np.uint8))
        assert np.array_equal(overlay[..., 1], np.zeros(shape, dtype=np.uint8))
        assert summary["overlays"][0]["missing_cases"] == ["caseB"]

    def test_sparse_overlay_precomputed_case_map_is_empty(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        for case in ("caseA", "caseB"):
            np.save(images / f"{case}.npy", np.ones(shape, dtype=np.float32))
        case_a_mask = masks / "caseA.npy"
        np.save(case_a_mask, np.ones(shape, dtype=np.uint8))

        _data, _meta, overlays, summary = _io.load_dir_collection(
            [str(images / "*.npy")],
            overlays=[
                (
                    "organ",
                    {"caseA": str(case_a_mask)},
                    True,
                    str(masks / "*.npy"),
                )
            ],
            case_regex=r"(?P<case>case[A-Z])",
        )

        overlay = overlays[0]["data"]
        assert np.array_equal(overlay[..., 0], np.ones(shape, dtype=np.uint8))
        assert np.array_equal(overlay[..., 1], np.zeros(shape, dtype=np.uint8))
        assert summary["overlays"][0]["pattern"] == str(masks / "*.npy")
        assert summary["overlays"][0]["missing_cases"] == ["caseB"]

    def test_sparse_overlay_ambiguous_layout_requests_case_regex(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        images = tmp_path / "images"
        masks = tmp_path / "masks"
        images.mkdir()
        masks.mkdir()
        for case in ("caseA", "caseB"):
            np.save(images / f"{case}.npy", np.ones(shape, dtype=np.float32))
        np.save(masks / "caseA.npy", np.ones(shape, dtype=np.uint8))

        with pytest.raises(ValueError, match="pass --case-regex"):
            _io.load_dir_collection(
                [str(images / "*.npy")],
                overlays=[("organ", str(masks / "*.npy"), True)],
            )

    def test_collection_reports_each_scanned_file(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case in ("caseA", "caseB"):
            np.save(tmp_path / f"{case}.npy", np.ones(shape, dtype=np.float32))
        scanned = []

        _io.load_dir_collection(
            [str(tmp_path / "*.npy")],
            scan_progress=lambda label, path: scanned.append((label, path)),
        )

        assert [label for label, _path in scanned] == ["images", "images"]
        assert [os.path.basename(path) for _label, path in scanned] == [
            "caseA.npy",
            "caseB.npy",
        ]

    def test_per_case_directories_are_inferred_for_sparse_overlays(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case in ("patientA", "patientB"):
            (tmp_path / case / "images").mkdir(parents=True)
            (tmp_path / case / "masks").mkdir()
            np.save(
                tmp_path / case / "images" / "scan.npy",
                np.full(shape, 1 if case == "patientA" else 2, dtype=np.float32),
            )
        np.save(
            tmp_path / "patientB" / "masks" / "organ.npy",
            np.ones(shape, dtype=np.uint8),
        )

        data, _meta, overlays, summary = _io.load_dir_collection(
            [str(tmp_path / "*" / "images" / "scan.npy")],
            overlays=[("organ", str(tmp_path / "*" / "masks" / "organ.npy"), True)],
        )

        assert summary["cases"] == ["patientA", "patientB"]
        assert np.all(data[..., 0] == 1)
        assert np.all(data[..., 1] == 2)
        assert np.all(overlays[0]["data"][..., 0] == 0)
        assert np.all(overlays[0]["data"][..., 1] == 1)
        assert summary["overlays"][0]["missing_cases"] == ["patientA"]

    def test_ordered_pairing_without_case_regex(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case_idx, case in enumerate(("patient_001", "patient_002")):
            pdir = tmp_path / case
            pdir.mkdir()
            np.save(pdir / "T1.npy", np.full(shape, case_idx, dtype=np.float32))
            np.save(pdir / "T2.npy", np.full(shape, case_idx + 10, dtype=np.float32))
            np.save(pdir / "ground_truth.npy", np.full(shape, case_idx + 1, dtype=np.uint8))

        data, _meta, overlays, summary = _io.load_dir_collection(
            [
                str(tmp_path / "*" / "T1.npy"),
                str(tmp_path / "*" / "T2.npy"),
            ],
            overlays=[("gt", str(tmp_path / "*" / "ground_truth.npy"))],
        )

        assert data.shape == (4, 5, 6, 2, 2)
        assert overlays[0]["data"].shape == (4, 5, 6, 2)
        assert summary["cases"] == ["patient_001", "patient_002"]
        assert np.array_equal(data[:, :, :, 1, 1], np.full(shape, 11, dtype=np.float32))
        assert np.array_equal(overlays[0]["data"][:, :, :, 1], np.full(shape, 2, dtype=np.uint8))

    def test_ragged_shapes_use_lazy_collection(self, tmp_path):
        import arrayview._io as _io
        from arrayview import _session as _session_mod
        from arrayview._analysis import _build_metadata

        cases = {
            "patient_001": ((4, 5, 6), (4, 6, 6)),
            "patient_002": ((3, 5, 6), (3, 6, 6)),
        }
        for case_idx, (case, (t1_shape, t2_shape)) in enumerate(cases.items()):
            pdir = tmp_path / case
            pdir.mkdir()
            np.save(pdir / "T1.npy", np.full(t1_shape, case_idx, dtype=np.float32))
            np.save(pdir / "T2.npy", np.full(t2_shape, case_idx + 10, dtype=np.float32))
            np.save(pdir / "ground_truth.npy", np.ones(t1_shape, dtype=np.uint8))

        data, _meta, overlays, summary = _io.load_dir_collection(
            [
                str(tmp_path / "*" / "T1.npy"),
                str(tmp_path / "*" / "T2.npy"),
            ],
            overlays=[("gt", str(tmp_path / "*" / "ground_truth.npy"))],
        )

        assert isinstance(data, _io._RaggedFileSeries)
        assert data.shape == (4, 6, 6, 2, 2)
        assert data.ragged_spatial_shapes == [
            [(4, 5, 6), (4, 6, 6)],
            [(3, 5, 6), (3, 6, 6)],
        ]
        assert data[:, :, :, 1, 0].shape == (3, 5, 6)
        assert data[:, :, 99, 1, 0].shape == (3, 5)
        assert overlays[0]["data"].shape == (4, 5, 6, 2)
        assert summary["cases"] == ["patient_001", "patient_002"]

        session = _session_mod.Session(data, filepath=None, name="ragged")
        session.collection_spatial_ndim = 3
        meta = _build_metadata(session)
        assert meta["shape"] == [4, 5, 6, 2, 2]
        assert meta["collection_spatial_ndim"] == 3
        assert meta["ragged_spatial_shapes"][1][1] == [3, 6, 6]

    def test_metadata_ignores_case_axis_when_selecting_startup_plane(self):
        from arrayview._analysis import _build_metadata
        from arrayview._session import Session

        session = Session(np.zeros((48, 3, 224, 224, 5), dtype=np.float32))
        session.collection_spatial_ndim = 4

        assert _build_metadata(session)["default_dims"] == [2, 3]

    def test_ordered_pairing_mismatched_counts_error(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        for case in ("patient_001", "patient_002"):
            pdir = tmp_path / case
            pdir.mkdir()
            np.save(pdir / "T1.npy", np.zeros(shape))
        np.save(tmp_path / "patient_001" / "ground_truth.npy", np.zeros(shape, dtype=np.uint8))

        with pytest.raises(ValueError, match="missing case.*patient_002"):
            _io.load_dir_collection(
                [str(tmp_path / "*" / "T1.npy")],
                overlays=[("gt", str(tmp_path / "*" / "ground_truth.npy"))],
            )

    def test_case_regex_overrides_default_key(self, tmp_path):
        import arrayview._io as _io

        shape = (4, 5, 6)
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        np.save(tmp_path / "images" / "img_case-a_ch0.npy", np.zeros(shape))
        np.save(tmp_path / "labels" / "label_case-a.npy", np.ones(shape, dtype=np.uint8))

        data, _meta, overlays, summary = _io.load_dir_collection(
            [str(tmp_path / "images" / "img_*.npy")],
            overlays=[("label", str(tmp_path / "labels" / "label_*.npy"))],
            case_regex=r"(?:img|label)_(?P<case>case-[^_/.]+)",
        )

        assert data.shape == (4, 5, 6, 1)
        assert overlays[0]["data"].shape == (4, 5, 6, 1)
        assert summary["cases"] == ["case-a"]
