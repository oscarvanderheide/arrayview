"""PyTorch deep-learning integration for arrayview.

Provides:
- view_batch()        — browse a DataLoader / Dataset / batch in the viewer
- TrainingMonitor     — live training visualisation via handle.update()

All torch imports are lazy — this module is safe to import without PyTorch.
"""

from __future__ import annotations

import numpy as np

from arrayview._launcher import view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_ndarray(obj):
    """Convert a tensor-like object to numpy, or return as-is if already ndarray."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _extract_images(source, *, key=None):
    """Extract an ndarray of images from a batch (dict, tuple, tensor, ndarray)."""
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, dict):
        if key is not None:
            return _tensor_to_ndarray(source[key])
        best_key, best_size = None, -1
        for k, v in source.items():
            arr = _tensor_to_ndarray(v)
            if arr.size > best_size:
                best_key, best_size = k, arr.size
        return _tensor_to_ndarray(source[best_key])
    if isinstance(source, (tuple, list)):
        return _tensor_to_ndarray(source[0])
    if hasattr(source, "detach") or hasattr(source, "numpy"):
        return _tensor_to_ndarray(source)
    raise TypeError(
        f"Unsupported batch type: {type(source).__name__}. "
        "Expected ndarray, tensor, dict, tuple, or list."
    )


def _is_dataloader(obj):
    """Heuristic: has __iter__ and a 'dataset' attribute."""
    return hasattr(obj, "__iter__") and hasattr(obj, "dataset")


def _is_dataset(obj):
    """Heuristic: has __getitem__ and __len__ but no 'dataset' attribute."""
    if isinstance(obj, (dict, list, tuple, np.ndarray)):
        return False
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__") and not hasattr(obj, "dataset")


# ---------------------------------------------------------------------------
# view_batch
# ---------------------------------------------------------------------------

def view_batch(source, *, samples=None, overlay=None, key=None, **kwargs):
    """Open an arrayview window to browse a batch of images.

    Parameters
    ----------
    source : DataLoader, Dataset, dict, tuple, ndarray, or tensor
    samples : int, optional
        How many samples to show.
    overlay : str, optional
        Key name in a dict-batch to use as segmentation overlay.
    key : str, optional
        Key name in a dict-batch to view.
    **kwargs
        Forwarded to :func:`arrayview.view`.
    """
    if _is_dataloader(source):
        batch = next(iter(source))
    elif _is_dataset(source):
        n = samples if samples is not None else 16
        import random
        indices = random.sample(range(len(source)), min(n, len(source)))
        items = [source[i] for i in indices]
        if isinstance(items[0], dict):
            batch = {
                k: np.stack([_tensor_to_ndarray(item[k]) for item in items])
                for k in items[0]
            }
        else:
            batch = np.stack([_tensor_to_ndarray(item) for item in items])
    else:
        batch = source

    images = _extract_images(batch, key=key)
    if samples is not None and images.shape[0] > samples:
        images = images[:samples]

    if overlay is not None and isinstance(batch, dict):
        ov = _tensor_to_ndarray(batch[overlay])
        if samples is not None and ov.shape[0] > samples:
            ov = ov[:samples]
        kwargs["overlay"] = ov

    return view(images, **kwargs)
