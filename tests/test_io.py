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

