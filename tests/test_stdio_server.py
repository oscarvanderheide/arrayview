from arrayview._stdio_server import _coalesce_slice_pane_key


def test_qmri_pane_key_distinguishes_maps_with_same_axes():
    base = {"sid": "s1", "dim_x": 2, "dim_y": 3}
    t1 = {**base, "pane_key": "qmri:0"}
    t2 = {**base, "pane_key": "qmri:1"}
    assert _coalesce_slice_pane_key(t1) != _coalesce_slice_pane_key(t2)


def test_slice_pane_key_falls_back_to_sid_and_axes():
    msg = {"sid": "s1", "dim_x": 2, "dim_y": 3}
    assert _coalesce_slice_pane_key(msg) == ("s1", 2, 3)
