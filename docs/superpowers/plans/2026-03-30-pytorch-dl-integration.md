# PyTorch Deep Learning Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `TrainingMonitor`, `view_batch()`, and Shift+O overlay toggle to make arrayview frictionless for PyTorch medical imaging workflows.

**Architecture:** One new Python module (`_torch.py`) with two public APIs that wrap existing `view()` + `handle.update()`. One frontend change in `_viewer.html` for the Shift+O overlay visibility toggle. No server changes.

**Tech Stack:** Python (numpy, lazy torch import), JavaScript (viewer keyboard handler)

---

### Task 1: Overlay Visibility Toggle (Shift+O) — Frontend

**Files:**
- Modify: `src/arrayview/_viewer.html` (keyboard handler ~line 7639, overlay request params ~line 5438, help text ~line 1357)

- [ ] **Step 1: Add overlay visibility state variable**

Near the existing `overlayAlpha` declaration at line 1624 in `_viewer.html`, add:

```javascript
let _overlayVisibility = 'all';  // 'all' | 'none' | 0 | 1 | 2 ... (index into overlay_sid list)
```

- [ ] **Step 2: Add helper to get visible overlay SIDs**

Add this function near `showStatus()` (~line 6128):

```javascript
function _getVisibleOverlaySids() {
    if (!overlay_sid) return '';
    if (_overlayVisibility === 'all') return overlay_sid;
    if (_overlayVisibility === 'none') return '';
    // Individual mask index
    const sids = overlay_sid.split(',');
    const idx = _overlayVisibility;
    if (idx >= 0 && idx < sids.length) return sids[idx];
    return overlay_sid;
}

function _getVisibleOverlayColors() {
    if (!overlay_colors) return '';
    if (_overlayVisibility === 'all') return overlay_colors;
    if (_overlayVisibility === 'none') return '';
    const colors = overlay_colors.split(',');
    const idx = _overlayVisibility;
    if (idx >= 0 && idx < colors.length) return colors[idx];
    return overlay_colors;
}
```

- [ ] **Step 3: Replace overlay_sid in slice request params**

In all locations where overlay params are set on the request URLSearchParams (lines 5438-5440, 5526-5528, 10110-10112, 10489-10491, 10693-10695, 11168-11170), replace:

```javascript
if (overlay_sid) params.set('overlay_sid', overlay_sid);
if (overlay_sid && overlay_colors) params.set('overlay_colors', overlay_colors);
```

with:

```javascript
const _visSid = _getVisibleOverlaySids();
if (_visSid) params.set('overlay_sid', _visSid);
if (_visSid && overlay_colors) { const _visCol = _getVisibleOverlayColors(); if (_visCol) params.set('overlay_colors', _visCol); }
```

The `overlay_alpha` line stays unchanged — it still reads from `overlayAlpha`.

- [ ] **Step 4: Add Shift+O keyboard handler**

Replace the existing Shift+O handler (line 7639). The condition `(e.key === 'O' && e.shiftKey && !modKey)` currently opens the file picker. Change it so that **when overlays are active**, Shift+O cycles visibility instead, and when no overlays, it falls through to the picker.

Replace this block (~line 7639):

```javascript
} else if (((e.key === 'o' || e.key === 'O') && modKey) || (e.key === 'O' && e.shiftKey && !modKey)) {
```

with two separate blocks:

```javascript
} else if (e.key === 'O' && e.shiftKey && !modKey && overlay_sid) {
    // Cycle overlay visibility: all → none → mask0 → mask1 → ... → all
    const sids = overlay_sid.split(',');
    const nMasks = sids.length;
    if (_overlayVisibility === 'all') {
        _overlayVisibility = 'none';
    } else if (_overlayVisibility === 'none') {
        _overlayVisibility = nMasks > 1 ? 0 : 'all';
    } else {
        // Currently showing individual mask
        const next = _overlayVisibility + 1;
        _overlayVisibility = next >= nMasks ? 'all' : next;
    }
    // Status message
    if (_overlayVisibility === 'all') {
        showStatus('overlays: all');
    } else if (_overlayVisibility === 'none') {
        showStatus('overlays: off');
    } else {
        const colors = (overlay_colors || '').split(',');
        const colorHex = colors[_overlayVisibility] || '';
        const colorTag = colorHex ? ` (#${colorHex})` : '';
        showStatus(`overlay ${_overlayVisibility + 1}/${nMasks}${colorTag}`);
    }
    updateView();
} else if (((e.key === 'o' || e.key === 'O') && modKey) || (e.key === 'O' && e.shiftKey && !modKey)) {
```

This way, Shift+O with overlays active → toggle; Shift+O without overlays → falls through to picker as before.

- [ ] **Step 5: Update help text**

At line 1357, update the help row for Shift+O. Replace:

```html
<div class="help-row"><span class="key" id="open-shortcut-help">Cmd+O</span><span class="desc">open picker (also Ctrl+O, Shift+O) — instance sidebar with thumbnails when multiple arrays loaded; RAM guard blocks oversized .pt/.tif/.mat files; Space selects, Enter opens (1 sel) or compares (2–4 sel)</span></div>
```

with:

```html
<div class="help-row"><span class="key" id="open-shortcut-help">Cmd+O</span><span class="desc">open picker (also Ctrl+O, Shift+O) — instance sidebar with thumbnails when multiple arrays loaded; RAM guard blocks oversized .pt/.tif/.mat files; Space selects, Enter opens (1 sel) or compares (2–4 sel)</span></div>
<div class="help-row"><span class="key">Shift+O</span><span class="desc">cycle overlay visibility: all → off → individual masks (when overlays active)</span></div>
```

- [ ] **Step 6: Reset visibility on overlay change**

When a new overlay is loaded, reset `_overlayVisibility` to `'all'`. Find where `overlay_sid` is assigned (search for `overlay_sid =` in the JS). Add after each assignment:

```javascript
_overlayVisibility = 'all';
```

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: Shift+O cycles overlay visibility (all/off/individual masks)"
```

---

### Task 2: `view_batch()` — Core Implementation

**Files:**
- Create: `src/arrayview/_torch.py`
- Modify: `src/arrayview/__init__.py`
- Test: `tests/test_torch.py`

- [ ] **Step 1: Write tests for `view_batch()`**

Create `tests/test_torch.py`:

```python
"""Tests for arrayview._torch (PyTorch DL integration)."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# We mock torch throughout — tests should not require a real PyTorch install.


# ---------- _tensor_to_ndarray helper ----------

class FakeTensor:
    """Mimics a PyTorch tensor with .detach().cpu().numpy() chain."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self._arr


class TestTensorToNdarray:
    def test_numpy_passthrough(self):
        from arrayview._torch import _tensor_to_ndarray
        arr = np.zeros((4, 8, 8))
        assert _tensor_to_ndarray(arr) is arr

    def test_fake_tensor(self):
        from arrayview._torch import _tensor_to_ndarray
        t = FakeTensor(np.ones((2, 3)))
        result = _tensor_to_ndarray(t)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, 1.0)


# ---------- _extract_images helper ----------

class TestExtractImages:
    def test_ndarray_batch(self):
        from arrayview._torch import _extract_images
        batch = np.random.rand(4, 64, 64)
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_tensor_batch(self):
        from arrayview._torch import _extract_images
        batch = FakeTensor(np.random.rand(4, 64, 64))
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_dict_batch_auto_key(self):
        from arrayview._torch import _extract_images
        batch = {
            'label': np.zeros((4, 64, 64), dtype=np.int32),
            'image': np.random.rand(4, 1, 128, 128).astype(np.float32),
        }
        result = _extract_images(batch)
        # Should pick 'image' — largest by element count
        assert result.shape == (4, 1, 128, 128)

    def test_dict_batch_explicit_key(self):
        from arrayview._torch import _extract_images
        batch = {
            'image': np.random.rand(4, 128, 128).astype(np.float32),
            'label': np.zeros((4, 64, 64), dtype=np.int32),
        }
        result = _extract_images(batch, key='label')
        assert result.shape == (4, 64, 64)

    def test_tuple_batch(self):
        from arrayview._torch import _extract_images
        batch = (np.random.rand(4, 64, 64), np.zeros(4))
        result = _extract_images(batch)
        assert result.shape == (4, 64, 64)

    def test_unsupported_type(self):
        from arrayview._torch import _extract_images
        with pytest.raises(TypeError):
            _extract_images("not a batch")


# ---------- view_batch ----------

class TestViewBatch:
    @patch('arrayview._torch.view')
    def test_ndarray_batch(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        batch = np.random.rand(4, 64, 64)
        view_batch(batch)
        mock_view.assert_called_once()
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape == (4, 64, 64)

    @patch('arrayview._torch.view')
    def test_dict_batch_with_overlay(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()
        batch = {
            'image': np.random.rand(4, 64, 64).astype(np.float32),
            'label': np.ones((4, 64, 64), dtype=np.int32),
        }
        view_batch(batch, overlay='label')
        mock_view.assert_called_once()
        kwargs = mock_view.call_args[1]
        assert kwargs['overlay'].shape == (4, 64, 64)

    @patch('arrayview._torch.view')
    def test_dataloader(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()

        # Fake DataLoader — an iterable that yields batches
        batch = np.random.rand(8, 32, 32)

        class FakeLoader:
            def __iter__(self):
                return iter([batch])

        view_batch(FakeLoader())
        mock_view.assert_called_once()

    @patch('arrayview._torch.view')
    def test_dataset_with_samples(self, mock_view):
        from arrayview._torch import view_batch
        mock_view.return_value = MagicMock()

        class FakeDataset:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return np.random.rand(64, 64)

        view_batch(FakeDataset(), samples=4)
        mock_view.assert_called_once()
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape == (4, 64, 64)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_torch.py -v
```

Expected: `ModuleNotFoundError: No module named 'arrayview._torch'`

- [ ] **Step 3: Write `_torch.py` with `view_batch()` and helpers**

Create `src/arrayview/_torch.py`:

```python
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
    # duck-typed path: works for PyTorch, JAX, etc.
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


def _extract_images(source, *, key=None):
    """Extract an ndarray of images from a batch (dict, tuple, tensor, ndarray).

    Returns a numpy array with a leading batch dimension.
    """
    if isinstance(source, np.ndarray):
        return source

    # dict-like batch (e.g. {'image': tensor, 'label': tensor})
    if isinstance(source, dict):
        if key is not None:
            return _tensor_to_ndarray(source[key])
        # auto-detect: pick the key with the largest element count
        best_key, best_size = None, -1
        for k, v in source.items():
            arr = _tensor_to_ndarray(v)
            if arr.size > best_size:
                best_key, best_size = k, arr.size
        return _tensor_to_ndarray(source[best_key])

    # tuple/list batch (e.g. (images, labels))
    if isinstance(source, (tuple, list)):
        return _tensor_to_ndarray(source[0])

    # bare tensor
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
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__") and not hasattr(obj, "dataset")


# ---------------------------------------------------------------------------
# view_batch
# ---------------------------------------------------------------------------

def view_batch(source, *, samples=None, overlay=None, key=None, **kwargs):
    """Open an arrayview window to browse a batch of images.

    Parameters
    ----------
    source : DataLoader, Dataset, dict, tuple, ndarray, or tensor
        The batch to visualise.
        - *DataLoader*: grabs one batch via ``next(iter(source))``.
        - *Dataset*: indexes ``samples`` random items and stacks them.
        - *dict*: picks the key with the largest tensor (or uses *key*).
        - *tuple/list*: treats the first element as images.
        - *ndarray/tensor*: used directly (leading dim = batch).
    samples : int, optional
        How many samples to show.  *None* = full batch for DataLoaders,
        16 for Datasets.
    overlay : str, optional
        Key name in a dict-batch to use as segmentation overlay.
    key : str, optional
        Key name in a dict-batch to view.  If *None*, auto-detects the
        largest tensor.
    **kwargs
        Forwarded to :func:`arrayview.view`.

    Returns
    -------
    ViewHandle
    """
    # --- resolve source to a raw batch ---
    if _is_dataloader(source):
        batch = next(iter(source))
    elif _is_dataset(source):
        n = samples if samples is not None else 16
        import random
        indices = random.sample(range(len(source)), min(n, len(source)))
        items = [source[i] for i in indices]
        # stack: each item can be an array, tensor, or dict
        if isinstance(items[0], dict):
            batch = {
                k: np.stack([_tensor_to_ndarray(item[k]) for item in items])
                for k in items[0]
            }
        else:
            batch = np.stack([_tensor_to_ndarray(item) for item in items])
    else:
        batch = source

    # --- extract image array ---
    images = _extract_images(batch, key=key)

    # --- optional: limit samples ---
    if samples is not None and images.shape[0] > samples:
        images = images[:samples]

    # --- optional: overlay ---
    ov = None
    if overlay is not None and isinstance(batch, dict):
        ov = _tensor_to_ndarray(batch[overlay])
        if samples is not None and ov.shape[0] > samples:
            ov = ov[:samples]
        kwargs["overlay"] = ov

    return view(images, **kwargs)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_torch.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_torch.py tests/test_torch.py
git commit -m "feat: add view_batch() for browsing DataLoader/Dataset batches"
```

---

### Task 3: `TrainingMonitor` — Core Implementation

**Files:**
- Modify: `src/arrayview/_torch.py`
- Modify: `tests/test_torch.py`

- [ ] **Step 1: Write tests for `TrainingMonitor`**

Append to `tests/test_torch.py`:

```python
# ---------- TrainingMonitor ----------

class TestTrainingMonitor:
    @patch('arrayview._torch.view')
    def test_first_step_opens_viewer(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(
            input=np.random.rand(1, 64, 64),
            target=np.ones((1, 64, 64), dtype=np.int32),
            prediction=np.random.rand(1, 64, 64),
            epoch=0,
        )
        mock_view.assert_called_once()

    @patch('arrayview._torch.view')
    def test_skipped_epoch(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=5, samples=1)
        # epoch 0 — opens viewer
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=0)
        mock_view.assert_called_once()

        # epoch 1 — skipped (not divisible by 5, not epoch 0)
        handle.reset_mock()
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=1)
        handle.update.assert_not_called()

    @patch('arrayview._torch.view')
    def test_update_called_on_matching_epoch(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=5, samples=1)
        mon.step(input=np.zeros((1, 8, 8)), target=np.zeros((1, 8, 8)),
                 prediction=np.zeros((1, 8, 8)), epoch=0)

        # epoch 5 — should update
        mon.step(input=np.ones((1, 8, 8)), target=np.ones((1, 8, 8)),
                 prediction=np.ones((1, 8, 8)), epoch=5)
        assert handle.update.call_count == 3  # input, target, prediction handles

    @patch('arrayview._torch.view')
    def test_two_pane_mode(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle)

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(input=np.zeros((1, 8, 8)), prediction=np.zeros((1, 8, 8)), epoch=0)
        assert mock_view.call_count == 1
        # Should have been called with 2 arrays
        args = mock_view.call_args[0]
        assert len(args) == 2

    @patch('arrayview._torch.view')
    def test_single_pane_mode(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = handle

        mon = TrainingMonitor(every=1, samples=1)
        mon.step(prediction=np.zeros((1, 8, 8)), epoch=0)
        assert mock_view.call_count == 1
        args = mock_view.call_args[0]
        assert len(args) == 1

    @patch('arrayview._torch.view')
    def test_multiple_samples_stacked(self, mock_view):
        from arrayview._torch import TrainingMonitor
        handle = MagicMock()
        mock_view.return_value = (handle, handle, handle)

        mon = TrainingMonitor(every=1, samples=3)
        # Call step 3 times in same epoch → collects 3 samples
        for i in range(3):
            mon.step(
                input=np.full((1, 8, 8), i, dtype=np.float32),
                target=np.zeros((1, 8, 8), dtype=np.int32),
                prediction=np.full((1, 8, 8), i * 0.1, dtype=np.float32),
                epoch=0,
            )
        # view() called once, after samples collected, with stacked arrays
        assert mock_view.call_count == 1
        arr_arg = mock_view.call_args[0][0]
        assert arr_arg.shape[0] == 3  # 3 samples stacked
```

- [ ] **Step 2: Run tests — verify new tests fail**

```bash
pytest tests/test_torch.py::TestTrainingMonitor -v
```

Expected: `ImportError: cannot import name 'TrainingMonitor'`

- [ ] **Step 3: Implement `TrainingMonitor`**

Add to `src/arrayview/_torch.py`, after `view_batch()`:

```python
# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    """Live training visualisation — updates an arrayview window periodically.

    Parameters
    ----------
    every : int
        Update the viewer every *every* epochs (default 1).
    samples : int
        Number of validation samples to collect per epoch (default 1).
        If > 1, samples are stacked along a new leading dimension.
    overlay : bool
        If *True*, show target/prediction as overlays on the input image
        instead of side-by-side compare panes.
    """

    def __init__(self, *, every=1, samples=1, overlay=False):
        self.every = every
        self.samples = samples
        self.overlay = overlay
        self._handles = None       # tuple of ViewHandle, or single ViewHandle
        self._epoch_buf = {}       # {epoch: {'input': [...], 'target': [...], 'prediction': [...]}}
        self._last_epoch = None

    def step(self, *, input=None, target=None, prediction=None, epoch):
        """Record a validation sample.  Opens/updates the viewer as needed.

        Call this inside your validation loop.  On skipped epochs (not
        divisible by *every*, except epoch 0) the call is a no-op.

        Parameters
        ----------
        input, target, prediction : array-like, optional
            Tensors or ndarrays.  At least *prediction* must be provided.
        epoch : int
            Current epoch number.
        """
        if epoch != 0 and epoch % self.every != 0:
            return

        # Convert tensors
        arrays = {}
        if input is not None:
            arrays['input'] = _tensor_to_ndarray(input)
        if target is not None:
            arrays['target'] = _tensor_to_ndarray(target)
        if prediction is not None:
            arrays['prediction'] = _tensor_to_ndarray(prediction)

        if not arrays:
            return

        # New epoch → reset buffer
        if epoch != self._last_epoch:
            self._epoch_buf[epoch] = {k: [] for k in arrays}
            self._last_epoch = epoch

        buf = self._epoch_buf[epoch]
        for k, v in arrays.items():
            if k not in buf:
                buf[k] = []
            if len(buf[k]) < self.samples:
                buf[k].append(v)

        # Not enough samples yet → wait
        any_key = next(iter(buf))
        if len(buf[any_key]) < self.samples:
            return

        # Stack samples if > 1
        stacked = {}
        for k, arrs in buf.items():
            stacked[k] = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]

        # Build pane list
        panes = []
        names = []
        if 'input' in stacked:
            panes.append(stacked['input'])
            names.append('input')
        if 'target' in stacked:
            panes.append(stacked['target'])
            names.append('target')
        if 'prediction' in stacked:
            panes.append(stacked['prediction'])
            names.append('prediction')

        if self._handles is None:
            # First time — open the viewer
            kwargs = {'name': names}
            if self.overlay and 'input' in stacked:
                # overlay mode: input with target+prediction as overlays
                overlays = []
                if 'target' in stacked:
                    overlays.append(stacked['target'])
                if 'prediction' in stacked:
                    overlays.append(stacked['prediction'])
                if overlays:
                    kwargs['overlay'] = overlays if len(overlays) > 1 else overlays[0]
                result = view(stacked['input'], **kwargs)
                self._handles = (result,) if not isinstance(result, tuple) else result
            else:
                result = view(*panes, **kwargs)
                self._handles = (result,) if not isinstance(result, tuple) else result
        else:
            # Update existing handles
            if self.overlay and 'input' in stacked:
                # In overlay mode there's only one handle for the input pane.
                # We can't update overlays via handle.update() — just update the base.
                self._handles[0].update(stacked['input'])
            else:
                for handle, (k, arr) in zip(self._handles, stacked.items()):
                    handle.update(arr)

        # Clean up buffer for this epoch
        self._epoch_buf.pop(epoch, None)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_torch.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_torch.py tests/test_torch.py
git commit -m "feat: add TrainingMonitor for live training visualisation"
```

---

### Task 4: Exports and Final Wiring

**Files:**
- Modify: `src/arrayview/__init__.py`

- [ ] **Step 1: Add lazy exports for TrainingMonitor and view_batch**

In `src/arrayview/__init__.py`, add the imports. These should be lazy to avoid requiring torch at import time. Replace the file contents with:

```python
__version__ = "0.6.0"

from arrayview._launcher import arrayview, view, ViewHandle  # noqa: F401
from arrayview._session import zarr_chunk_preset  # noqa: F401
from arrayview._torch import TrainingMonitor, view_batch  # noqa: F401
```

Note: `_torch.py` itself only imports torch lazily (inside functions), so this direct import is safe — it won't trigger a torch import until the user actually calls `TrainingMonitor()` or `view_batch()`.

- [ ] **Step 2: Verify imports work without torch installed**

```bash
python -c "import arrayview; print(arrayview.__version__)"
```

Expected: prints `0.6.0` with no errors (torch is not imported at module load time).

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/test_torch.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/__init__.py
git commit -m "feat: export TrainingMonitor and view_batch from arrayview"
```

---

### Task 5: Update visual_smoke.py for Overlay Toggle

**Files:**
- Modify: `tests/visual_smoke.py`

- [ ] **Step 1: Check if visual_smoke.py has an overlay section**

Read `tests/visual_smoke.py` and look for existing overlay tests. If there's an overlay test scenario, add a Shift+O toggle check. If not, add a new scenario.

Add a test step to the overlay scenario (or create one) that:
1. Loads an array with an overlay
2. Presses Shift+O and verifies the status shows `"overlays: off"`
3. Presses Shift+O again and verifies the status shows `"overlays: all"`

The exact code depends on the existing patterns in `visual_smoke.py` — match them.

- [ ] **Step 2: Run visual smoke test**

```bash
pytest tests/visual_smoke.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/visual_smoke.py
git commit -m "test: add Shift+O overlay toggle to visual smoke tests"
```
