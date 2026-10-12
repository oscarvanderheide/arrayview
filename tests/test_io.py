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
