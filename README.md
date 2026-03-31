# arrayview

A viewer for multi-dimensional arrays.

- CLI and Python
- Jupyter / VS Code
- Browser / native
- SSH / tunnels

## CLI

```bash
uvx arrayview scan.nii.gz
uvx arrayview --window browser scan.npy
uvx arrayview                            # demo
```

## Python

```python
from arrayview import view
view(arr)
```

## MATLAB

Add the `matlab/` directory to your MATLAB path, then:

```matlab
addpath('/path/to/arrayview/matlab')

A = rand(100, 200, 10);
arrayview(A)
```

Requires arrayview installed in [MATLAB's Python environment](https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html):

```bash
pip install arrayview
```

Arrays are passed zero-copy via the buffer protocol (in-process Python). `arrayview()` enables this automatically — just call it before any other `py.*` call in your MATLAB session.

## PyTorch / Deep Learning

```python
from arrayview import view_batch, TrainingMonitor

# Browse a DataLoader batch
view_batch(train_loader)
view_batch(train_loader, overlay='label')

# Live training monitor — updates every N epochs
monitor = TrainingMonitor(every=5, samples=3)
for epoch in range(100):
    for batch in val_loader:
        pred = model(batch['image'])
        monitor.step(input=batch['image'], target=batch['label'],
                     prediction=pred, epoch=epoch)
```

`view_batch()` accepts DataLoaders, Datasets, dicts, tuples, or raw tensors. `TrainingMonitor` opens a compare window and calls `handle.update()` automatically. PyTorch is not required at import time.


## Formats

`.npy` `.npz` `.nii` `.nii.gz` `.zarr` `.pt` `.h5` `.tif` `.mat`

## Once open

`c` colormaps · `d` dynamic range · `v` 3-plane · `z` mosaic · `Shift+O` overlay toggle · `?` help · colorbar dblclick histogram


## nnInteractive Segmentation

`S` starts AI-assisted 3D segmentation (requires CUDA). Click/draw to segment, `Enter` to accept.

```toml
[nninteractive]
url = "http://gpu-server:1527"   # skip auto-launch, use running server
```

Or: `ARRAYVIEW_NNINTERACTIVE_URL=http://gpu-server:1527`


## Config

`~/.arrayview/config.toml`:

```toml
[viewer]
colormaps = ["gray", "viridis", "plasma"]   # colormaps cycled by 'c'

[window]
default = "browser"                         # browser | native | vscode | inline
```

[Full documentation →](https://oscarvanderheide.github.io/arrayview/)
