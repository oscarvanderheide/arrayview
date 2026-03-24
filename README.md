# arrayview

A fast, minimal viewer for multi-dimensional arrays, inspired by the great
[arrShow](https://github.com/tsumpf/arrShow) and [sigpy's ImagePlot](https://github.com/mikgroup/sigpy/blob/5da0e8605f166be41e520ef0ef913482487611d8/sigpy/plot.py#L48).

- CLI and Python usage
- Inline in Jupyter / VS Code interactive window
- Native desktop window or browser
- Runs locally, over SSH, and through VS Code tunnels
- Julia support via PythonCall

## Install into a Python project

```bash
uv add arrayview      # or: pip install arrayview
```

## Quick Start

### CLI

```bash
uvx arrayview scan.nii.gz          # native window (default)
uvx arrayview --window browser scan.npy   # open in browser
uvx arrayview                       # animated demo (no file needed)
```

### Python

```python
from arrayview import view
import numpy as np

x = np.random.rand(256, 256, 32)
view(x)
```

### Jupyter / VS Code Notebook

```python
from arrayview import view
import numpy as np

x = np.random.rand(256, 256, 32)
view(x)  # inline IFrame (auto-detected in Jupyter)
```

### Julia (via PythonCall)

```julia
using PythonCall
arrayview = pyimport("arrayview")
x = rand(Float32, 256, 256, 32)
arrayview.view(x)
```

## Supported File Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.npy` | numpy | Memory-mapped (no full load into RAM) |
| `.npz` | numpy | Must contain exactly one array |
| `.nii` / `.nii.gz` | nibabel | Lazy proxy (loaded on demand) |
| `.zarr` / `.zarr.zip` | zarr | Read mode |
| `.pt` / `.pth` | torch | `weights_only=True`, converted to numpy |
| `.h5` / `.hdf5` | h5py | Must contain exactly one dataset |
| `.tif` / `.tiff` | tifffile | Full load |
| `.mat` | scipy | Must contain exactly one ndarray |

Optional libraries (nibabel, zarr, torch, h5py, tifffile, scipy) are imported
only when the corresponding format is opened.

## Python API

### `view(data, ...)`

```python
view(
    data,               # array-like (numpy, PyTorch, JAX, Julia, Zarr, nibabel, h5py)
    name=None,          # display name (default: "Array {shape}")
    port=8123,          # server port (auto-scans if busy)
    inline=None,        # None=auto, True=force IFrame, False=no inline
    height=500,         # IFrame height in pixels
    window=None,        # see below
    rgb=False,          # treat as RGB/RGBA (first or last dim must be 3 or 4)
)
```

`view()` returns a `ViewHandle` — a string subclass that behaves as the viewer URL and additionally exposes a `.update(arr)` method for pushing new data without reopening a window:

```python
v = view(arr)          # opens viewer, v == "http://localhost:8123/?sid=..."
v.update(arr2)         # viewer refreshes in-place with new array
v.update(arr3)         # update again — keeps the same window/tab open
print(v.sid)           # session ID
print(v.port)          # server port
```

**`window` modes:**

| Value | Behavior |
|-------|----------|
| `None` | Auto: native window outside Jupyter, inline IFrame inside |
| `True` | Native window (browser fallback if unavailable) |
| `False` | No auto-open; returns URL |
| `"native"` | Native desktop window |
| `"browser"` | System browser |
| `"vscode"` | VS Code Simple Browser |
| `"inline"` | Inline IFrame |

### `zarr_chunk_preset(shape)`

Returns a recommended Zarr chunk shape optimized for interactive slice navigation:

```python
from arrayview import zarr_chunk_preset

chunks = zarr_chunk_preset((512, 512, 200, 10))
# (512, 512, 1, 2)  -- full XY tile, one Z-slice, 2 T-frames
```

## Zarr Support

[Zarr](https://zarr.readthedocs.io) is a chunked, compressed array format designed
for large datasets that don't fit in RAM. ArrayView opens `.zarr` stores lazily —
only the slices you actually navigate are loaded from disk or object storage.

### Convert a numpy array to Zarr

```python
import numpy as np
import zarr
from arrayview import zarr_chunk_preset

arr = np.random.rand(512, 512, 200)   # e.g. a 3D MRI volume

chunks = zarr_chunk_preset(arr.shape)  # (512, 512, 1) — one Z-slice per chunk
zarr.save_array("scan.zarr", arr, chunks=chunks, compressor=zarr.Blosc(cname="zstd"))
```

The chunk shape matters for navigation speed: the viewer reads one chunk per
slice, so a chunk that spans the full XY plane and one step along the scroll
axis loads the minimum data per interaction.

`zarr_chunk_preset` picks these shapes automatically based on the number of
dimensions:

| ndim | Example shape | Chunk shape | Notes |
|------|---------------|-------------|-------|
| 2 | `(H, W)` | `(H, W)` | Whole array in one chunk |
| 3 | `(H, W, Z)` | `(H, W, 1)` | One Z-slice per chunk |
| 4 | `(H, W, Z, T)` | `(H, W, 1, 2)` | One Z-slice, 2 T-frames |
| 5 | `(H, W, Z, T, C)` | `(H, W, 1, 1, C)` | Full channel axis per chunk |

XY tile size is capped at 1024×1024 for very large spatial dims.

### Open a Zarr array in arrayview

```python
from arrayview import view
import zarr

arr = zarr.open("scan.zarr", mode="r")
view(arr)                  # lazy — slices load on demand
```

Or from the CLI:

```bash
arrayview scan.zarr
arrayview scan.zarr.zip    # compressed zip store also supported
```

## CLI Reference

```
arrayview [FILES...] [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `FILES` | Array paths. First is the base; extras (up to 6) preload for compare mode. |
| `--port PORT` | Server port (default: 8000, auto-scans if busy) |
| `--window {browser,vscode,native}` | How to open the viewer |
| `--serve` | Start a persistent empty server (useful for VS Code tunnel setup) |
| `--kill` | Kill the server on `--port` and exit |
| `--overlay FILE` | Segmentation mask overlay (binary 0/1, same spatial shape) |
| `--vectorfield FILE` | Deformation vector field overlay (one axis of size 3 for xyz components) |
| `--vectorfield-components-dim DIM` | Axis index of the xyz component dimension in `--vectorfield` |
| `--rgb` | Interpret as RGB/RGBA |
| `--compare FILE` | Deprecated: use positional args instead |
| `--browser` | Deprecated: use `--window browser` |
| `--verbose` | Verbose internal output |

**No arguments** launches an animated RGB plasma demo.

### Compare Mode (CLI)

```bash
arrayview base.npy moving.npy                    # 2-way compare
arrayview base.npy moving.npy registered.npy     # 3-way compare
arrayview scan.nii.gz --overlay mask.nii.gz      # overlay a segmentation mask
```

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| Scroll wheel | Previous / next slice |
| `h` `l` / Left Right | Move cursor to previous / next dimension |
| `j` `k` / Down Up | Previous / next index along active dimension |
| `r` | Reverse active axis direction; in multi-view: swap axes globally across all panes |
| Space | Toggle auto-play |
| `+` / `-` / `0` | Zoom in / out / fit to window |

### Axes and Views

| Key | Action |
|-----|--------|
| `x` | Swap horizontal dim with slice dim |
| `y` | Swap vertical dim with slice dim |
| `t` | Transpose x and y axes |
| `z` | Claim dim as z (mosaic grid) |
| `v` | Toggle 3-plane multi-view (dims 0,1,2) |
| `V` | Toggle 3-plane multi-view (custom dims) |
| `o` | Reset oblique slices and crosshair to center |
| `q` | Toggle qMRI mode |

### Display

| Key | Action |
|-----|--------|
| `c` | Cycle colormap (gray, lipari, navia, viridis, plasma, RdBu_r, twilight_shifted) |
| `C` | Enter custom matplotlib colormap name |
| `d` | Cycle dynamic range (0-100%, 1-99%, 5-95%, 10-90%) |
| `D` | Set vmin / vmax manually |
| `L` | Toggle log scale |
| `m` | Cycle complex mode (mag / phase / real / imag) |
| `f` | Toggle centered FFT |
| `M` | Cycle mask threshold (Otsu + 7 levels) |
| `R` | Toggle RGB mode |
| `T` | Cycle theme (dark / light / solarized / nord) |
| `b` | Toggle canvas border |
| `F` | Zen mode — hide chrome, go fullscreen |
| `K` | Compact mode — collapse top chrome, vertical colorbar (auto-activates when zoomed) |
| `[` / `]` | Adjust arrow density in vectorfield mode |
| `{` / `}` | Adjust arrow length in vectorfield mode |
| `U` | Toggle vector arrows |

### Compare Mode

| Key | Action |
|-----|--------|
| `P` | Open picker (compare) |
| `O` | Open picker (open / navigate sessions) |
| `n` | Cycle compare target |
| `X` | Cycle center pane (off / A−B / \|A−B\| / \|A−B\|/\|A\| / overlay / wipe) |
| `[` / `]` | Decrease / increase overlay blend |
| `Z` | Focus center pane |

### Info and Export

| Key | Action |
|-----|--------|
| Mouse hover | Show pixel value on colorbar |
| `H` | Toggle pixel hover tooltip |
| `i` | Show data info (shape, dtype, size, path) |
| `s` | Save screenshot (PNG) |
| `g` | Save GIF of current slice dimension |
| `e` | Copy reusable URL to clipboard |
| `?` | Toggle help overlay |

### Mouse

| Input | Action |
|-------|--------|
| Drag on colorbar | Shift window level |
| Scroll on colorbar | Zoom window range |
| Double-click colorbar | Reset window/level |
| Shift+drag (3-plane) | Oblique rotation |
| Drag (3-plane) | Move crosshair |

## Viewing Modes

| Mode | Trigger | Description |
|------|---------|-------------|
| **Normal** | Default | Single 2D slice with dimension navigation |
| **Multi-view** | `v` / `V` | 3-plane orthogonal view (axial/coronal/sagittal) with oblique rotation |
| **Compare** | `P` (picker) | Side-by-side comparison of up to 6 arrays |
| **Center pane** | `X` (in compare) | Difference (A−B, \|A−B\|, \|A−B\|/\|A\|), overlay, or wipe |
| **qMRI** | `q` | Quantitative MRI: auto-detects parameter dimension, shows each with a dedicated colormap |

## VS Code Integration

ArrayView auto-detects VS Code terminals and opens in Simple Browser.

**Local VS Code terminal:** Works automatically. A bundled extension handles
the open request.

**VS Code tunnel / remote SSH:**

```bash
# 1. Start a persistent server on the remote machine
arrayview --serve

# 2. In VS Code Ports tab: right-click port 8000 → Port Visibility → Public

# 3. Load arrays freely — each opens in Simple Browser on your local VS Code
arrayview scan.nii.gz
arrayview base.npy moving.npy
```

The server persists across invocations so you only do steps 1–2 once per
session. Kill it with `arrayview --kill` when done.

**VS Code tunnel + server (multi-hop):**

Use this when your working data lives on a server that you SSH into from
the tunnel-remote machine.

```
Local VS Code ──(devtunnel)──▶ remote machine ──(SSH)──▶ server
```

1. On the remote machine: start a persistent server and set port 8000 to Public
   (same as above).

2. SSH from the remote machine to the server with a **reverse tunnel**:
   ```bash
   ssh -R 8000:localhost:8000 user@gpu-server
   ```
   This forwards server port 8000 → remote machine port 8000 (where
   ArrayView is running).

3. On the server, load your array normally — ArrayView detects the reverse
   tunnel automatically:
   ```bash
   arrayview array.npy
   ```
   The array bytes are sent to the remote server and the viewer opens in Simple
   Browser on your local VS Code just like in the direct tunnel case.

   > If port 8000 is occupied on the GPU server by something else, use a
   > different local port:
   > ```bash
   > ssh -R 8765:localhost:8000 user@gpu-server
   > arrayview array.npy --relay 8765
   > ```

## SSH (without VS Code)

ArrayView prints a port-forwarding hint when it detects an SSH session:

```
ssh -L 8000:localhost:8000 user@remote
```

Then open `http://localhost:8000` in your local browser.

## Development

```bash
git clone https://github.com/oscarvanderheide/arrayview
cd arrayview
uv sync --group test
uv run playwright install chromium
```

Run the tests:

```bash
uv run pytest tests/              # all (API + browser)
uv run pytest tests/test_api.py   # HTTP layer only
uv run pytest tests/test_browser.py  # Playwright/Chromium only
```

Visual regression baselines live in `tests/snapshots/`. Delete a file there to
reset its baseline.
