"""Preferences → Reopen where you left off: per-file view storage."""

import os

import numpy as np

import arrayview._view_memory as vm


class _Session:
    def __init__(self, path, shape):
        self.filepath = str(path)
        self.shape = shape
        self.array_key = None


def test_view_round_trip_and_staleness(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "VIEW_MEMORY_PATH", str(tmp_path / "view_memory.json"))
    f = tmp_path / "a.npy"
    np.save(f, np.zeros((4, 5)))
    s = _Session(f, (4, 5))
    assert vm.remembered_view(s) is None
    assert vm.remember_view(s, {"indices": [1, 2], "manualVmin": 0.0, "manualVmax": 1.0})
    assert vm.remembered_view(s) == {"indices": [1, 2], "manualVmin": 0.0, "manualVmax": 1.0}

    # Same shape, new contents: keep the slice, drop the range.
    np.save(f, np.ones((4, 5)))
    os.utime(f, ns=(1, 1))
    assert vm.remembered_view(s) == {"indices": [1, 2]}

    # Different shape: nothing comes back.
    assert vm.remembered_view(_Session(f, (5, 4))) is None
