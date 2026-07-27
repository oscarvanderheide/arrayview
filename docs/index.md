# arrayview

A viewer for multi-dimensional arrays.

- CLI and Python
- Jupyter / VS Code
- Browser / native
- SSH / tunnels

## CLI

```bash
uvx arrayview scan.nii.gz
uvx arrayview volume.npy
uvx arrayview                    # interactive tutorial
```

Starting without a file opens a generated 4-D lesson. It teaches navigation,
colormaps, histograms, orthogonal view, playback, comparisons, stacks,
overlays, ROI analysis, preferences, and the main ways to launch ArrayView.
The lesson advances only after you perform each action; sections can be
skipped, restarted, or resumed after a reload.

## Python

```python
from arrayview import view
import numpy as np

view(np.random.rand(256, 256, 32))
```

Works in scripts, Jupyter notebooks, and VS Code interactive windows.

MATLAB and Julia setup: [MATLAB and Julia](foreign-hosts.md).

Remote and tunnel setup: [Remote](remote.md).

Stacked patient collections and masks: [Stack and Overlay Collections](stack-overlays.md).

## Formats

`.npy` `.npz` `.nii` `.nii.gz` `.zarr` `.pt` `.h5` `.tif` `.mat`

Optional libraries (nibabel, zarr, torch, h5py, tifffile, scipy) are imported only when needed.

## Once open

`c` colormaps · `d` histogram · `v` 3-plane · `z` mosaic · `?` help
