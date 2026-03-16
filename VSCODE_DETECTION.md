# VS Code Terminal Detection — How It Works

## Problem

When running `arrayview` or `view()` from a VS Code integrated terminal, we want to open in VS Code Simple Browser, not native window. This requires detecting VS Code terminal environment and forcing browser opening mode.

## Root Cause

The default behavior when `window=None` (the default) is:
1. If not in Jupyter → set `window=True`
2. If `window=True` and native window available → open native window
3. This ignores VS Code detection

## Solution (as of commit XXXXXX)

In `view()` function in `_launcher.py`, after the initial window mode resolution (lines 579-585), we added VS Code terminal auto-detection (lines 587-591):

```python
# Auto-detect VS Code terminal: prefer Simple Browser over native window
if _in_vscode_terminal() and not _force_vscode and not _force_browser:
    _force_vscode = True
    if window is True:  # Convert bool to False so we hit the browser path below
        window = False
```

This ensures that when running in VS Code terminal:
- `_force_vscode` is set to `True`
- `window` is converted from `True` to `False`
- The code path at lines 766-793 will then call `_open_browser(url_viewer, force_vscode=True)`
- Which calls `_open_via_signal_file()` and installs the VS Code extension

## Detection Chain

`_in_vscode_terminal()` checks (in `_platform.py`):
1. `TERM_PROGRAM=vscode` environment variable
2. `VSCODE_IPC_HOOK_CLI` environment variable
3. Walk parent processes to find `VSCODE_IPC_HOOK_CLI` (handles `uv run` env stripping)

## Testing

Test in VS Code integrated terminal:
```bash
# Python script
python -c "from arrayview import view; import numpy as np; view(np.random.rand(100, 100))"

# CLI
uv run arrayview file.npy

# Should open in VS Code Simple Browser, not native window
```

## Related Files

- `src/arrayview/_launcher.py` — main `view()` function with detection logic
- `src/arrayview/_platform.py` — `_in_vscode_terminal()` detection function
- `src/arrayview/_vscode.py` — `_open_via_signal_file()` and extension install

## Tunnel (Remote) Enter Prompt Fix

### Problem

When running `uvx --from git+... arrayview file.npy` on a tunnel remote, the Enter prompt at line ~1814 calls `input()` which can receive `EOFError` immediately (stdin at EOF from the way uvx/shell launches the process). The old code did:

```python
except (EOFError, KeyboardInterrupt):
    sys.exit(0)  # BUG: exits before _open_browser is called
```

This caused the signal file to never be written → Simple Browser never opened → `ERR_CONNECTION_REFUSED`.

### Fix (current code)

Split EOFError and KeyboardInterrupt handling:

```python
except KeyboardInterrupt:
    sys.exit(0)  # intentional abort: user pressed Ctrl+C
except EOFError:
    pass  # stdin was at EOF: proceed immediately to _open_browser
```

`_open_browser` is now always called, even when stdin provides no input.
