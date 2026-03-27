# Loading Arrays

## Supported Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.npy` | numpy | Memory-mapped |
| `.npz` | numpy | Must contain one array |
| `.nii` / `.nii.gz` | nibabel | Lazy proxy |
| `.zarr` / `.zarr.zip` | zarr | Chunked access |
| `.pt` / `.pth` | torch | Converted to numpy |
| `.h5` / `.hdf5` | h5py | Must contain one dataset |
| `.tif` / `.tiff` | tifffile | Full load |
| `.mat` | scipy | Must contain one ndarray |

Optional libraries are imported only when needed.

## CLI

```bash
uvx arrayview volume.nii.gz
uvx arrayview volume.npy --window browser
uvx arrayview image.npy --rgb
uvx arrayview --watch data.npy              # reload on file change
```

## Python

```python
from arrayview import view
import numpy as np

x = np.random.rand(256, 256, 32)
v = view(x)
```

`view()` returns a `ViewHandle`:

```python
v = view(arr)
v.update(arr2)        # refresh without reopening
print(v.sid)          # session ID
print(v.port)         # server port
```

Key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | array-like | The array to display |
| `name` | str | Label shown in the viewer tab |
| `port` | int | Server port (default 8123) |
| `window` | str \| None | How to open the viewer (see below) |
| `rgb` | bool | Treat last/first axis as RGB/RGBA channels |
| `overlay` | array or list | Arrays composited as overlays |

`window` values:

| Value | Behaviour |
|-------|-----------|
| `None` | Auto: native outside Jupyter, inline inside |
| `"native"` | Native desktop window |
| `"browser"` | System browser |
| `"vscode"` | VS Code Simple Browser |
| `"inline"` | Inline IFrame (Jupyter / VS Code notebook) |

## File Picker

`Cmd/Ctrl+O` or `P` opens the file picker.

## Drag and Drop

Drop a file onto the viewer.

## Multiple Arrays

```bash
uvx arrayview base.npy moving.npy           # compare mode
uvx arrayview volume.nii.gz --overlay mask.nii.gz
```

## Zarr

Use `zarr_chunk_preset` to get chunk shapes optimized for slice navigation:

```python
from arrayview import zarr_chunk_preset

chunks = zarr_chunk_preset((512, 512, 200, 10))
# (512, 512, 1, 2)
```

Recommended chunk shapes by dimensionality:

| ndim | Axes | Chunk pattern |
|------|------|---------------|
| 2 | Y, X | `(Y, X)` |
| 3 | Y, X, Z | `(Y, X, 1)` |
| 4 | Y, X, Z, T | `(Y, X, 1, t)` |
| 5 | Y, X, Z, T, C | `(Y, X, 1, 1, C)` |

XY tile size is capped at 512.

Open a zarr store directly:

```bash
uvx arrayview scan.zarr
```
