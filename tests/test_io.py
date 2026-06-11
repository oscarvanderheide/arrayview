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


def test_stdio_mat_register_exposes_keys_and_reload_switches_array(tmp_path, monkeypatch):
    import scipy.io

    import arrayview._stdio_server as stdio
    from arrayview._session import SESSIONS

    path = tmp_path / "stdio_multiple_numeric.mat"
    scipy.io.savemat(
        path,
        {
            "first": np.zeros((2, 3), dtype=np.float32),
            "second": np.ones((4, 5), dtype=np.float32),
            "metadata": np.array([{"name": "not-displayable"}], dtype=object),
        },
    )

    responses = []
    monkeypatch.setattr(stdio, "_write_json", responses.append)

    before = set(SESSIONS)
    try:
        stdio._handle_register({"path": str(path)})
        assert len(responses) == 1
        sid = responses[0]["sid"]
        assert responses[0]["shape"] == [2, 3]
        assert SESSIONS[sid].array_keys == [
            {"key": "first", "shape": [2, 3], "dtype": "float32"},
            {"key": "second", "shape": [4, 5], "dtype": "float32"},
        ]
        assert "error" not in responses[0]

        responses.clear()
        stdio._handle_fetch_proxy(
            {
                "endpoint": f"/session/{sid}/reload-key",
                "method": "POST",
                "body": '{"key":"second"}',
            }
        )

        assert responses == [{"ok": True}]
        assert SESSIONS[sid].shape == (4, 5)
    finally:
        for sid in set(SESSIONS) - before:
            SESSIONS.pop(sid, None)


def test_stdio_npz_register_exposes_keys_and_reload_switches_array(tmp_path, monkeypatch):
    import arrayview._stdio_server as stdio
    from arrayview._session import SESSIONS

    path = tmp_path / "stdio_multiple_arrays.npz"
    np.savez(
        path,
        first=np.zeros((2, 3), dtype=np.float32),
        second=np.ones((4, 5), dtype=np.float32),
    )

    responses = []
    monkeypatch.setattr(stdio, "_write_json", responses.append)

    before = set(SESSIONS)
    try:
        stdio._handle_register({"path": str(path)})
        sid = responses[0]["sid"]
        assert responses[0]["shape"] == [2, 3]
        assert SESSIONS[sid].array_keys == [
            {"key": "first", "shape": [2, 3], "dtype": "float32"},
            {"key": "second", "shape": [4, 5], "dtype": "float32"},
        ]

        responses.clear()
        stdio._handle_fetch_proxy(
            {
                "endpoint": f"/session/{sid}/reload-key",
                "method": "POST",
                "body": '{"key":"second"}',
            }
        )

        assert responses == [{"ok": True}]
        assert SESSIONS[sid].shape == (4, 5)
    finally:
        for sid in set(SESSIONS) - before:
            SESSIONS.pop(sid, None)
