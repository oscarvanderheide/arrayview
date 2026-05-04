"""VS Code direct shared-memory transport — bypasses file I/O for large arrays."""

from __future__ import annotations

import atexit
import os
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker

import numpy as np

from arrayview._vscode_signal import _write_vscode_signal, _VSCODE_SIGNAL_MAX_AGE_MS

# Shared memory blocks kept alive until process exit or subprocess reads them.
_ACTIVE_SHM: list = []

def _open_direct_via_shm(
    data: "np.ndarray",
    name: str = "array",
    title: str | None = None,
    floating: bool = False,
) -> bool:
    """Write a direct-mode signal file with shared memory parameters.

    Places the array in POSIX shared memory so the extension-spawned subprocess
    can read it without any disk I/O.
    """
    import atexit
    import sys as _sys
    from multiprocessing.shared_memory import SharedMemory

    import numpy as np

    arr = np.ascontiguousarray(data)
    shm = SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr

    # The subprocess will unlink the SHM after reading it.  Unregister from
    # Python's resource tracker so it doesn't warn about "leaked" SHM at exit.
    from multiprocessing import resource_tracker
    try:
        resource_tracker.unregister(f"/{shm.name}", "shared_memory")
    except Exception:
        pass

    # Keep the shm alive until the subprocess reads it (or this process exits).
    _ACTIVE_SHM.append(shm)

    def _cleanup():
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)

    payload: dict = {
        "action": "open-preview",
        "mode": "direct",
        "shm": {
            "name": shm.name,
            "shape": ",".join(str(int(s)) for s in arr.shape),
            "dtype": str(arr.dtype),
        },
        "arrayName": name,
        "pythonPath": _sys.executable,
        "maxAgeMs": _VSCODE_SIGNAL_MAX_AGE_MS,
    }
    if title:
        payload["title"] = title
    if floating:
        payload["floating"] = True
    return _write_vscode_signal(payload, skip_compat=True)
