---
name: invocation-consistency
description: Use when implementing any server-side, startup, display-opening, or environment-detection feature in arrayview. Ensures the feature works correctly across all six ways arrayview can be launched — CLI, Python script, Jupyter, Julia, VS Code tunnel, and SSH.
---

# ArrayView Invocation Consistency Checklist

## Rule

Every behavior that depends on *how* arrayview is started (server lifecycle, browser opening, display routing, port forwarding) must be verified across all six invocation paths before it is considered done.

## The Six Invocation Paths

| Path | Entry point | Display | Server model |
|------|------------|---------|-------------|
| **CLI** | `arrayview()` | native window → browser fallback | background thread, blocks until Ctrl-C |
| **Python script** | `view(arr)` | native window → browser fallback | daemon thread (dies with caller) |
| **Jupyter** | `view(arr)` | inline IFrame | daemon thread (persists across cells) |
| **Julia (PythonCall)** | `view(arr)` | system browser | always subprocess (`_view_julia()`) — GIL |
| **VS Code terminal** | `arrayview()` or `view()` | VS Code Simple Browser | in-process or subprocess |
| **Plain SSH (no VS Code)** | `arrayview()` or `view()` | prints port-forward hint | in-process or subprocess |

## Key Detection Functions (all in `_platform.py`)

```python
_in_jupyter()           # ipykernel present → display inline IFrame
_in_vscode_terminal()   # TERM_PROGRAM=vscode OR VSCODE_IPC_HOOK_CLI → use Simple Browser
_is_vscode_remote()     # tunnel/SSH remote with VS Code server → shared signal fallback
_is_julia_env()         # juliacall in sys.modules or julia in sys.executable → subprocess
_can_native_window()    # pywebview available + not remote + display present → native window
_find_vscode_ipc_hook() # walk process tree for VSCODE_IPC_HOOK_CLI (stripped by uv run)
```

## Detection Order in `view()`

Julia → Jupyter → VS Code remote → VS Code terminal → local (native/browser)

Any new detection must not break the fallback chain for paths it shouldn't match.

## Julia-Specific Constraints

- Never run server in-process when `_is_julia_env()` is True (GIL conflicts)
- `_view_julia()` starts a detached subprocess; array is serialized to a temp `.npy` file
- No interactive stdin in subprocesses; all params encoded in CLI flags or files

## Jupyter-Specific Constraints

- `_in_jupyter()` returns True for ipykernel (VS Code notebook, JupyterLab, classic Notebook)
- Julia's IJulia kernel is NOT ipykernel → `_in_jupyter()` returns False in Julia notebooks
- `inline=True` → returns `IPython.display.IFrame`; caller must return it from the cell
- Port reuse: repeated `view()` calls reuse same port/server if already running

## Port & URL Rules

- Always use `localhost` (not `127.0.0.1`) so VS Code port-forwarding works
- Default port: `8123`; CLI: `--port` flag; `view()`: `port=` kwarg

## tmux and VS Code Terminal Detection

tmux breaks the ancestor-walk for `VSCODE_IPC_HOOK_CLI` because the process's parent chain goes through `tmux-server` (an independent daemon with no VS Code env), not through the VS Code terminal shell.

**Why `tmux show-environment` fails:** `VSCODE_IPC_HOOK_CLI` is not in tmux's default `update-environment` list.

**Why `#{client_pid}` alone fails:** returns only one client; breaks with multiple attached clients.

**Correct approach — enumerate all clients for the current session:**

```python
session_id = subprocess.run(["tmux", "display-message", "-p", "#{session_id}"], ...).stdout.strip()
client_pids = subprocess.run(
    ["tmux", "list-clients", "-t", session_id, "-F", "#{client_pid}"], ...
).stdout.strip().splitlines()
for pid_str in client_pids:
    val = _ipc_from_pid(int(pid_str))
    if val and os.path.exists(val):
        return val
```

`list-clients -t <session_id>` scopes to current session only — prevents false-positive from a different VS Code window in another tmux session.

## Step-by-Step Checklist

1. **Server startup/teardown?**
   - CLI: starts and terminates cleanly
   - Python script: daemon thread doesn't orphan
   - Jupyter: repeated calls don't fail on port-already-in-use
   - Julia: `_view_julia()` passes new params via CLI flag or file

2. **Browser/display opening?**
   - Local native window (macOS): `pywebview` opens correctly
   - VS Code terminal: Simple Browser opens via extension signal file
   - VS Code tunnel: reaches client-side Simple Browser, not remote host
   - Verify `_ensure_vscode_extension()` installs/updates if VSIX changed

3. **Environment detection?**
   - Check detection order: Julia → Jupyter → VS Code remote → VS Code terminal → local
   - `_find_vscode_ipc_hook()` walks up to 12 parent processes; new subprocess wrappers may need the same

4. **VS Code extension changed?**
   - Rebuild VSIX: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`
   - Bump `_VSCODE_EXT_VERSION` in `_vscode.py` and `vscode-extension/package.json` together

## Quick Automated Checks

```bash
uv run pytest tests/test_api.py -x
uv run pytest tests/test_cli.py -x
uv run python -c "from arrayview import view; import numpy as np; view(np.zeros((10,10)))"
```

## Red Flags — STOP

- "I changed server startup but only tested CLI" → test all paths
- "Works locally but not in tunnel" → check `_is_vscode_remote()` path and port exposure
- "I used `127.0.0.1` in the URL" → use `localhost`
- "I fixed tmux with `#{client_pid}`" → only works with one client; use `list-clients`
- "I bumped `package.json` but not `_VSCODE_EXT_VERSION`" → must stay in sync
- "Server starts but leaves an orphan after Ctrl-C" → check `_shutdown_event` and subprocess reaping
