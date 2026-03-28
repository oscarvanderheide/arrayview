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

## Formats

`.npy` `.npz` `.nii` `.nii.gz` `.zarr` `.pt` `.h5` `.tif` `.mat`

## Once open

`c` colormaps · `d` dynamic range · `v` 3-plane · `z` mosaic · `?` help · colorbar dblclick histogram

## Config

`~/.arrayview/config.toml`:

```toml
[viewer]
colormaps = ["gray", "viridis", "plasma"]   # colormaps cycled by 'c'

[window]
default = "browser"                         # browser | native | vscode | inline
```

[Full documentation →](https://oscarvanderheide.github.io/arrayview/)
