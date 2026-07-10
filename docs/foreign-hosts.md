# MATLAB and Julia

ArrayView uses the Python environment selected by the host application.

## MATLAB

Select a Python environment with ArrayView installed, then import and call it:

```matlab
pyenv(Version="/path/to/python")
py.importlib.import_module("arrayview");

h = py.arrayview.view(A);
```

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

Julia currently supports one array per `view()` call. Call `view()` again for
another array.

ArrayView copies the array to a temporary NumPy file and starts or reuses a
separate Python server process. This avoids blocking on Julia's Python lock and
lets the viewer survive after the call returns. The returned value is a Python
`ViewHandle` wrapper with `url`, `sid`, `port`, `update()`, and `close()`.

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
| VS Code remote or tunnel | VS Code tab through a forwarded port |
| Jupyter or IJulia | Inline iframe |
| Plain SSH | Browser URL through SSH port forwarding |

For plain SSH, follow [Remote](remote.md#ssh). VS Code tunnel forwarding is
handled by the extension.

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
