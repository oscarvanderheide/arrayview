# MATLAB and Julia

ArrayView uses the Python environment selected by the host application.

## MATLAB

Select a Python environment with ArrayView installed and put
[`matlab/arrayview.m`](https://github.com/oscarvanderheide/arrayview/blob/main/matlab/arrayview.m)
on the MATLAB path:

```matlab
pyenv(Version="/path/to/python")

arrayview(A)                                   % one array
arrayview(A, B)                                % side by side
arrayview(A, B, 'overlay', mask, 'cmap', 'viridis', 'voxel_size', [1 1 3])
[ha, hb] = arrayview(A, B);                    % one handle per array
hb.update(B2)                                  % replaces only the second pane
```

It takes the same options as Python's `view()`: `name`, `overlay`, `rgb`,
`mode`, `dims`, `index`, `cmap`, `vmin`, `vmax`, `log`, `complex`,
`voxel_size`, `window` and `port`. `dims` and `index` count from 1, as MATLAB
does. `E` in the viewer copies the current view as an `arrayview(...)` line.

Without the wrapper, call Python directly: `h = py.arrayview.view(A);`

Local MATLAB follows the normal display selection and prefers a native window
when available. MATLAB started through SSH or VS Code follows that environment's
browser or VS Code route.

The array is converted to NumPy before registration. Keep `h` when explicit
cleanup is needed:

```matlab
h.close()
```

## Julia

PythonCall:

```julia
using PythonCall

arrayview = pyimport("arrayview")
h = arrayview.view(A)
```

PyCall:

```julia
using PyCall

arrayview = pyimport("arrayview")
h = arrayview.view(A)
```

Everything Python's `view()` takes works from Julia too: several arrays side
by side, overlays, colour images and the view options.

```julia
ha, hb = arrayview.view(A, B; overlay=mask, cmap="viridis", voxel_size=(1, 1, 3))
hb.update(B2)        # replaces only the second pane
```

`dims` and `index` count from 0 here, because this is the Python function.
Several arrays in one call must have the same number of dimensions (their
sizes may differ) and cannot be colour images.

ArrayView copies the arrays to temporary NumPy files and starts or reuses a
separate Python server process. This avoids blocking on Julia's Python lock and
lets the viewer survive after the call returns. The returned value is a Python
`ViewHandle` wrapper with `url`, `sid`, `port`, `update()`, and `close()`; a
several-array call returns one per array.

```julia
h.close()
```

## IJulia

Use the same PythonCall or PyCall code in an IJulia notebook. ArrayView displays
an inline iframe by default. Inline display is a side effect; the call may return
`nothing` instead of a handle.

## Display routing

| Host | Default display |
|------|-----------------|
| Local desktop | Native window when available, otherwise browser |
| VS Code terminal | VS Code tab |
| VS Code remote or tunnel | VS Code tab through VS Code's private proxy |
| Jupyter or IJulia | Inline iframe |
| Plain SSH | Browser URL through SSH port forwarding |

For plain SSH, follow [Remote](remote.md#ssh). VS Code tunnel proxying is
handled privately by the extension; no Public entry in the Ports view is
required.

## Cleanup

In Python, close a handle directly or use a context manager:

```python
handle = arrayview.view(a)
handle.close()

with arrayview.view(a) as handle:
    print(handle.url)
```

`close()` releases that viewer session. Calling it again is a no-op. If cleanup
fails, it raises an error and can be retried.

## Advanced server setup

Start with `view(A)`. Use a persistent server only for shared or multi-hop
setups:

```bash
arrayview --serve
arrayview data.npy
```

Use `--relay PORT` only when a reverse SSH tunnel exposes ArrayView on a
different remote port. See [Remote](remote.md#multi-hop).
