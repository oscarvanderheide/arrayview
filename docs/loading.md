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
| `.mat` | scipy / h5py | Multi-array: in-viewer picker for array selection |

Optional libraries are imported only when needed.

Multi-array formats (`.npz`, `.mat`) show an in-viewer picker when they contain more than one array.

## CLI

```bash
uvx arrayview volume.nii.gz
uvx arrayview volume.npy --window browser
uvx arrayview image.npy --rgb
uvx arrayview --watch data.npy              # reload on file change
uvx arrayview --version                     # print version
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
| `port` | int | Server port (default 8123 for Python API, 8000 for CLI) |
| `window` | str \| None | How to open the viewer (see below) |
| `rgb` | bool | Treat last/first axis as RGB/RGBA channels |
| `overlay` | array or list | Arrays composited as overlays |

`window` values:

| Value | Behaviour |
|-------|-----------|
| `None` | Auto: native outside Jupyter, inline inside |
| `"native"` | Native desktop window |
| `"browser"` | System browser |
| `"vscode"` | VS Code tab |
| `"inline"` | Inline IFrame (Jupyter / VS Code notebook) |

## File Picker

`Cmd/Ctrl+O` or `P` opens the file picker.

## Drag and Drop

Drop a file onto the viewer.

## Startup

The first rendered frame is shown immediately. There is no client-side intro animation.
Native windows also show a static preview while the live viewer warms up.

## Multiple Arrays

```bash
uvx arrayview base.npy moving.npy           # compare mode
uvx arrayview volume.nii.gz --overlay mask.nii.gz
```

## NIfTI Series (4D/5D from a directory)

Stack a directory of NIfTI files into a single lazy 4D/5D array — only the
viewed slice is loaded, so RAM stays bounded regardless of series size.

```bash
uvx arrayview patients/ --stack-nifti
```

Discovers `.nii`/`.nii.gz` recursively, groups by immediate parent folder
(= patient), and ignores nested NIfTI folders once the parent has series files.
One file per patient → 4D `(*vol, P)`. The viewer opens with X/Y on screen,
Z as primary scroll, patient index as a slider.

Multiple files per patient (e.g. `t1`, `t2`, `flair`) → use `--select`:

```bash
uvx arrayview patients/ --stack-nifti --select '*t1*' --select '*t2*' --select '*flair*'
```

Each `--select` pattern picks one file per patient (fnmatch on basename).
Produces 5D `(*vol, P, M)` with modality as the last axis. Every patient
must match exactly one file per pattern.

Python API:

```python
from arrayview import view_dir

view_dir("patients/")
view_dir("patients/", select=["*t1*", "*t2*", "*flair*"])
```

Patient folders with no NIfTI files (e.g. only `.dcm`) raise an error —
convert DICOM to NIfTI first (e.g. `dcm2niix`).

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
