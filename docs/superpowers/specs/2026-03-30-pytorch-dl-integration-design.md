# PyTorch Deep Learning Integration

**Date:** 2026-03-30
**Status:** Approved

## Goal

Make arrayview frictionless for medical imaging researchers using PyTorch, by adding two convenience features that wrap the existing `view()` + `handle.update()` API.

## Scope

- One new file: `src/arrayview/_torch.py`
- Exports added to `__init__.py`: `TrainingMonitor`, `view_batch`
- No server changes, no frontend changes
- PyTorch remains a lazy import (not a hard dependency)

## Feature 1: `TrainingMonitor`

A class that opens an arrayview compare window and updates it periodically during training.

### API

```python
from arrayview import TrainingMonitor

monitor = TrainingMonitor(
    every=5,          # update viewer every N epochs (default: 1)
    samples=3,        # how many samples to keep from validation (default: 1)
    overlay=False,    # if True, show prediction as overlay on input instead of side-by-side
)
```

### Usage in a training loop

```python
for epoch in range(100):
    train(model, train_loader)

    for batch in val_loader:
        pred = model(batch['image'])
        monitor.step(
            input=batch['image'],
            target=batch['label'],
            prediction=pred,
            epoch=epoch,
        )
```

### Behavior

- **First call to `step()`**: calls `view(input, target, prediction)` to open a 3-pane compare window. Stores the returned `ViewHandle` objects.
- **Subsequent calls within the same epoch**: replaces the stored samples if `samples` limit not yet reached. If `samples=3`, keeps the first 3 calls per epoch.
- **Every N epochs** (controlled by `every`): calls `handle.update()` on the prediction pane to refresh the viewer. Input and target panes are also updated (the user may be viewing different validation samples each time).
- **Skipped epochs** (not divisible by `every`): `step()` is a no-op (returns immediately). Cheap to leave in the loop.
- **Tensor handling**: all tensors are detached, moved to CPU, and converted to numpy internally. The user never needs to do `.detach().cpu().numpy()`.

### Data shape handling

- **Single sample** (no batch dim or batch=1): displayed directly.
- **Multiple samples** (`samples > 1`): stacked along a new leading dimension so the user can scroll through them in arrayview's existing dimension slider.
- **2D or 3D spatial data**: both work — arrayview handles them natively.
- **Segmentation labels** (integer tensors): when `overlay=True`, passed via `view()`'s existing `overlay=` parameter for label colorization.

### What gets shown

Three options depending on arguments provided to `step()`:

| Arguments provided | Panes shown |
|---|---|
| `input`, `target`, `prediction` | 3-pane compare: input / target / prediction |
| `input`, `prediction` | 2-pane compare: input / prediction |
| `prediction` only | 1-pane: just the prediction |

When `overlay=True` and `target`/`prediction` are integer masks:
- Instead of compare mode, shows `input` with `target` as overlay in pane 1, `input` with `prediction` as overlay in pane 2.

## Feature 2: `view_batch()`

A convenience function to quickly browse samples from a DataLoader, Dataset, or raw batch.

### API

```python
from arrayview import view_batch

# From a DataLoader — grabs one batch
view_batch(train_loader)

# From a Dataset — grabs N random samples
view_batch(dataset, samples=16)

# From a raw batch (dict or tuple)
batch = next(iter(train_loader))
view_batch(batch)

# With segmentation overlay
view_batch(train_loader, overlay='label')
```

### Parameters

- `source`: a `DataLoader`, `Dataset`, dict, tuple, or tensor
- `samples` (int, default=None): how many samples to show. None = full batch for DataLoaders, 16 for Datasets.
- `overlay` (str, default=None): key name in a dict-batch to use as segmentation overlay.
- `key` (str, default=None): if the batch is a dict, which key to view. If None, uses heuristic: picks the largest tensor (by element count), which is usually the image.
- All other kwargs forwarded to `view()`.

### Batch format detection

| Input type | Behavior |
|---|---|
| `DataLoader` | `next(iter(source))` to grab one batch |
| `Dataset` | Index `samples` random items, stack them |
| `dict` | Look up `key` (or auto-detect largest tensor) |
| `tuple/list` | First element is treated as images |
| `Tensor/ndarray` | Used directly (assumed to be [N, ...] batch) |

### Output

- Stacks samples along a new leading dimension → single (N+d)-D array.
- Opens with `view()` — user scrolls through samples with existing dimension slider.
- If `overlay` is specified, the overlay array is stacked the same way and passed to `view(overlay=...)`.
- Returns the `ViewHandle` so the user can further interact programmatically.

## Module structure: `_torch.py`

```
src/arrayview/_torch.py
├── TrainingMonitor        (class)
│   ├── __init__()         — store config
│   ├── step()             — main entry point called in training loop
│   ├── _should_update()   — check if this epoch triggers an update
│   ├── _collect_sample()  — detach/cpu/numpy a tensor
│   └── _refresh_viewer()  — call view() or handle.update()
│
├── view_batch()           (function)
│   ├── _extract_from_loader()
│   ├── _extract_from_dataset()
│   ├── _extract_from_dict()
│   └── _detect_image_key()
```

## Exports

Add to `src/arrayview/__init__.py`:

```python
from arrayview._torch import TrainingMonitor, view_batch
```

These imports must be lazy — `_torch.py` imports `torch` only when its functions are actually called, not at module load time. This keeps `import arrayview` fast and doesn't require PyTorch to be installed.

## Error handling

- If PyTorch is not installed and user calls `TrainingMonitor()` or `view_batch()`: raise `ImportError` with a clear message ("pip install torch").
- If batch format is unrecognized: raise `TypeError` with explanation of supported formats.
- If tensor is on GPU: silently move to CPU (no warning needed, this is expected).

## Testing

- Unit tests for tensor conversion (detach/cpu/numpy path).
- Unit tests for batch format detection (dict, tuple, DataLoader, Dataset).
- Integration test: `TrainingMonitor.step()` with mock tensors, verify `view()` and `handle.update()` are called correctly.
- No need to test the viewer itself — that's already covered.

## Feature 3: Overlay Visibility Toggle (Shift+O)

A keyboard shortcut to cycle through overlay visibility states in the viewer.

### Current state

- `[`/`]` adjusts overlay alpha (transparency) — already works
- Shift+O opens file picker (redundant with Cmd+O / Ctrl+O)
- No way to hide overlays or view individual masks

### New behavior: Shift+O cycles through

1. **All overlays shown** (default, current behavior)
2. **No overlays** (base image only)
3. **Mask 1 only** (if multiple overlays)
4. **Mask 2 only**
5. ...etc for each mask
6. Back to all

When cycling, show a status message like `"overlays: all"`, `"overlays: off"`, `"overlay 1/3: red"`.

If only one overlay is present, the cycle is just: **shown → hidden → shown**.

### Implementation

- Frontend only (`_viewer.html`): add keyboard handler for Shift+O
- Track `overlayVisibility` state: `'all'` | `'none'` | index (0, 1, 2, ...)
- When rendering, filter which `overlay_sid` values are sent in the slice request based on visibility state
- No server changes needed — the server already supports rendering with any subset of overlays

### Why this matters for DL integration

With `TrainingMonitor(overlay=True)`, researchers get ground truth and prediction as separate overlay masks. Shift+O lets them quickly compare: just the prediction, just the ground truth, both, or neither.

## Out of scope

- PyTorch Lightning / other framework callbacks (future wrapper around `TrainingMonitor`).
- Feature map / activation visualization.
- Loss curves / scalar metrics.
- Attention maps / GradCAM.
