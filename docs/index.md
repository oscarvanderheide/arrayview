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
uvx arrayview                    # demo
```

## Python

```python
from arrayview import view
import numpy as np

view(np.random.rand(256, 256, 32))
```

Works in scripts, Jupyter notebooks, and VS Code interactive windows.

## Formats

`.npy` `.npz` `.nii` `.nii.gz` `.zarr` `.pt` `.h5` `.tif` `.mat`

Optional libraries (nibabel, zarr, torch, h5py, tifffile, scipy) are imported only when needed.

## Once open

`c` colormaps · `d` histogram · `v` 3-plane · `z` mosaic · `?` help
