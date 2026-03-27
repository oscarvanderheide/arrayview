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

## Formats

`.npy` `.npz` `.nii` `.nii.gz` `.zarr` `.pt` `.h5` `.tif` `.mat`

## Once open

`c` colormaps · `d` histogram · `v` 3-plane · `z` mosaic · `?` help

[Full documentation →](https://oscarvanderheide.github.io/arrayview/)
