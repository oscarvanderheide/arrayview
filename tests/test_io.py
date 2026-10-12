import numpy as np


def test_mat_loader_ignores_non_numeric_arrays_when_selecting_default(tmp_path):
    import scipy.io

    from arrayview._io import default_array_key, list_array_keys, load_data

    path = tmp_path / "with_metadata.mat"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    scipy.io.savemat(
        path,
        {
            "image": expected,
            "metadata": np.array([{"name": "not-displayable"}], dtype=object),
        },
    )

    assert list_array_keys(str(path)) == [
        {"key": "image", "shape": [3, 4], "dtype": "float32"}
    ]
    assert default_array_key(str(path)) == "image"
    np.testing.assert_array_equal(load_data(str(path)), expected)


def test_mat_loader_returns_key_list_for_multiple_numeric_arrays(tmp_path):
    import pytest
    import scipy.io

    from arrayview._io import default_array_key, list_array_keys, load_data

    path = tmp_path / "multiple_numeric.mat"
    scipy.io.savemat(
        path,
        {
            "first": np.zeros((2, 3), dtype=np.float32),
            "second": np.ones((4, 5), dtype=np.uint16),
            "metadata": np.array([{"name": "not-displayable"}], dtype=object),
        },
    )

    assert list_array_keys(str(path)) == [
        {"key": "first", "shape": [2, 3], "dtype": "float32"},
        {"key": "second", "shape": [4, 5], "dtype": "uint16"},
    ]
    assert default_array_key(str(path)) == "first"
    with pytest.raises(ValueError, match=r"\.mat file contains multiple arrays"):
        load_data(str(path))
    np.testing.assert_array_equal(
        load_data(str(path), key="second"),
        np.ones((4, 5), dtype=np.uint16),
    )



def _write_gipl(path, xyz, image_type, opener=open):
    header = bytearray(256)
    dims = list(xyz.shape) + [1] * (4 - xyz.ndim)
    header[0:8] = np.array(dims, ">u2").tobytes()
    header[8:10] = np.array([image_type], ">u2").tobytes()
    header[252:256] = np.array([4026526128], ">u4").tobytes()
    with opener(path, "wb") as f:
        f.write(bytes(header) + xyz.T.astype(xyz.dtype.newbyteorder(">")).tobytes())


def test_gipl_loader_reads_x_y_z_order_plain_and_gzipped(tmp_path):
    import gzip

    from arrayview._io import _SUPPORTED_EXTS, _get_ext, load_data

    expected = np.arange(4 * 5 * 3, dtype=np.int16).reshape(4, 5, 3) - 30
    _write_gipl(tmp_path / "vol.gipl", expected, 15)
    _write_gipl(tmp_path / "vol.gipl.gz", expected, 15, opener=gzip.open)

    for name in ("vol.gipl", "vol.gipl.gz"):
        path = str(tmp_path / name)
        assert _get_ext(path) in _SUPPORTED_EXTS
        loaded = load_data(path)
        assert loaded.dtype == np.int16
        np.testing.assert_array_equal(loaded, expected)


def test_gipl_loader_rejects_non_gipl_file(tmp_path):
    import pytest

    from arrayview._io import load_data

    path = tmp_path / "bogus.gipl"
    path.write_bytes(bytes(300))
    with pytest.raises(ValueError, match="not a GIPL file"):
        load_data(str(path))


def _write_mat_v73(path, arrays):
    import h5py

    with h5py.File(path, "w", userblock_size=512) as f:
        for name, value in arrays.items():
            f.create_dataset(name, data=value)
    header = b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM"
    with open(path, "r+b") as fh:
        fh.write(header)


def test_mat_v73_contiguous_array_is_mapped_not_read_whole(tmp_path):
    from arrayview._io import load_data

    path = tmp_path / "v73.mat"
    # Trailing pair is the largest, so the viewer starts on one contiguous block.
    expected = np.arange(3 * 4 * 32 * 40, dtype=np.float32).reshape(3, 4, 32, 40)
    _write_mat_v73(path, {"parmaps": expected})

    loaded = load_data(str(path))
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)


def test_mat_v73_complex_maps_to_complex64(tmp_path):
    from arrayview._io import load_data

    path = tmp_path / "v73_complex.mat"
    values = np.arange(8 * 32 * 40, dtype=np.float32).reshape(8, 32, 40)
    compound = np.empty(values.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
    compound["real"] = values
    compound["imag"] = -values
    _write_mat_v73(path, {"sense": compound})

    loaded = load_data(str(path))
    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, values - 1j * values)


def test_mat_v73_compressed_array_still_loads(tmp_path):
    import h5py

    from arrayview._io import load_data

    path = tmp_path / "v73_gz.mat"
    expected = np.arange(2 * 32 * 40, dtype=np.float32).reshape(2, 32, 40)
    with h5py.File(path, "w", userblock_size=512) as f:
        f.create_dataset("x", data=expected, compression="gzip")
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM")

    loaded = load_data(str(path))
    assert not isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)


def test_read_npy_into_ram_matches_np_load(tmp_path, monkeypatch):
    import pytest

    from arrayview import _io

    # Tiny chunks so every array spans many parallel reads.
    monkeypatch.setattr(_io, "_NPY_READ_CHUNK_BYTES", 4096)
    rng = np.random.default_rng(0)
    cases = {
        "c.npy": rng.random((7, 33, 41), dtype=np.float32),
        "f.npy": np.asfortranarray(rng.random((7, 33, 41))),
        "be.npy": rng.random((5, 300)).astype(">f4"),
        "cx.npy": (rng.random((3, 70, 50)) + 1j).astype(np.complex64),
        "empty.npy": np.zeros((0, 4), dtype=np.int16),
    }
    for name, expected in cases.items():
        path = tmp_path / name
        np.save(path, expected)
        mapped = np.load(path, mmap_mode="r")
        seen = []
        loaded = _io.read_npy_into_ram(mapped, lambda d, t: seen.append((d, t)))
        assert not isinstance(loaded, np.memmap)
        assert loaded.dtype == expected.dtype
        assert loaded.flags.f_contiguous == mapped.flags.f_contiguous
        np.testing.assert_array_equal(loaded, expected)
        if expected.nbytes:
            assert seen[-1] == (expected.nbytes, expected.nbytes)
    with pytest.raises(OSError):
        path = tmp_path / "truncated.npy"
        np.save(path, np.zeros((64, 64), dtype=np.float64))
        mapped = np.load(path, mmap_mode="r")
        with open(path, "r+b") as fh:
            fh.truncate(path.stat().st_size - 100)
        _io.read_npy_into_ram(mapped)


def test_viewer_start_index_mirrors_viewer_rule():
    from arrayview._io import _viewer_start_index

    # Displayed and comparably sized axes start in the middle; small ones at 0.
    assert _viewer_start_index((256, 256, 192, 4, 4), (0, 1)) == (128, 128, 96, 0, 0)
    assert _viewer_start_index((5, 204, 240, 224), (2, 3)) == (0, 102, 120, 112)


def test_fast_first_frame_maps_large_c_order_npy_on_local_disk(tmp_path, monkeypatch):
    from arrayview import _io

    path = tmp_path / "c_order.npy"
    # One element per page along the viewer's starting axes: not cheap to map.
    expected = np.arange(48 * 40 * 1100, dtype=np.float32).reshape(48, 40, 1100)
    np.save(path, expected)

    monkeypatch.setattr(_io, "_on_known_local_disk", lambda _p: True)
    loaded = _io.load_data(str(path), fast_first_frame=True)
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)

    # Off a local disk, or without the flag, it is read whole as before.
    assert not isinstance(_io.load_data(str(path)), np.memmap)
    monkeypatch.setattr(_io, "_on_known_local_disk", lambda _p: False)
    assert not isinstance(_io.load_data(str(path), fast_first_frame=True), np.memmap)
