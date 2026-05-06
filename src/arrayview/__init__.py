__version__ = "0.20.0"

__all__ = [
    "TrainingMonitor",
    "ViewHandle",
    "arrayview",
    "view",
    "view_batch",
    "zarr_chunk_preset",
]


def __getattr__(name: str):
    if name in {"arrayview", "view", "ViewHandle"}:
        from arrayview import _launcher

        return getattr(_launcher, name)
    if name == "zarr_chunk_preset":
        from arrayview._session import zarr_chunk_preset

        return zarr_chunk_preset
    if name in {"TrainingMonitor", "view_batch"}:
        from arrayview import _torch

        return getattr(_torch, name)
    raise AttributeError(f"module 'arrayview' has no attribute {name!r}")
