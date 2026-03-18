---
name: invocation-consistency
description: Use when implementing any server-side, startup, display-opening, or environment-detection feature in arrayview. Ensures the feature works correctly across all six ways arrayview can be launched — CLI, Python script, Jupyter, Julia, VS Code tunnel, and SSH.
---

# ArrayView Invocation Consistency Checklist

## Rule

Every behavior that depends on *how* arrayview is started (server lifecycle, browser opening, display routing, port forwarding) must be verified across all six invocation paths before it is considered done.

## The Six Invocation Paths

| Path | Entry point | Key detection | Server model |
|------|------------|--------------|-------------|
| **CLI** | `arrayview()` in `_app.py` | — (always CLI path) | `_run_server_subprocess()` or in-process thread |
| **Python script** | `view(arr)` | `_in_jupyter()` → False | In-process daemon thread |
| **Jupyter / VS Code interactive** | `view(arr)` | `_in_jupyter()` → True | In-process daemon thread |
| **Julia (PythonCall)** | `view(arr)` | `_is_julia_env()` → True | Always subprocess (`_view_julia()`) |
| **VS Code tunnel / SSH remote** | `arrayview()` or `view()` | `_is_vscode_remote()` → True | Subprocess server |
| **Plain SSH (no VS Code)** | `arrayview()` or `view()` | `SSH_CONNECTION` set, no VS Code remote | In-process or subprocess; prints port-forward hint |

## Key Detection Functions (all in `_app.py`)

```python
_in_jupyter()           # ipykernel present → display inline IFrame
_in_vscode_terminal()   # TERM_PROGRAM=vscode OR VSCODE_IPC_HOOK_CLI → use Simple Browser
_is_vscode_remote()     # tunnel/SSH remote with VS Code server → subprocess + extension routing
_is_julia_env()         # juliacall in sys.modules or julia in sys.executable → subprocess
_can_native_window()    # pywebview available + not remote + display present → native window
_find_vscode_ipc_hook() # walk process tree for VSCODE_IPC_HOOK_CLI (stripped by uv run)
```

## Feature Categories & Where Each Path Diverges

### Server Lifecycle

| Path | Server model | Implication |
|------|-------------|-------------|
| CLI | Background thread in `main()` → blocks until Ctrl-C | `_shutdown_event` drives cleanup |
| Python script | `_start_server_thread()` → daemon thread | Dies with caller process |
| Jupyter | `_start_server_thread()` → daemon thread | Persists across cells; port reuse matters |
| Julia | `_view_julia()` spawns detached subprocess | Caller (Julia) may not share GIL; subprocess manages its own lifecycle |
| VS Code remote | Subprocess via `_run_server_subprocess()` | Caller exits; server stays alive independently |
| Plain SSH | In-process or subprocess | No GUI; user must port-forward manually |

When changing server startup or shutdown logic, check whether daemonized threads, subprocess reaping, and port reuse are all handled correctly for each path.

### Browser / Display Opening

| Path | Opens how | Key function |
|------|-----------|-------------|
| CLI local | Native window (`pywebview`) OR system browser | `_open_webview_with_fallback()` → `_open_browser()` |
| Python script local | Same as CLI local | Same |
| Jupyter | Inline IFrame via `IPython.display.IFrame` | `view()` returns IFrame object |
| Julia | System browser via subprocess signal | `_view_julia()` writes signal file |
| VS Code terminal (local) | VS Code Simple Browser via extension | `_open_via_signal_file()` + `_ensure_vscode_extension()` |
| VS Code tunnel/remote | VS Code Simple Browser on *client* side | signal file + extension + port auto-forward |
| Plain SSH | Prints `ssh -L <port>:localhost:<port>` hint | `_open_browser()` fallback |

When changing URL construction, port logic, or display routing, trace through `_open_browser()` and its callers for each path — particularly the VS Code tunnel path where the *client's* VS Code picks up the signal file.

### VS Code Extension (`arrayview-opener.vsix`)

- Extension is auto-installed by `_ensure_vscode_extension()` when `_in_vscode_terminal()` is True
- Version must match `_VSCODE_EXT_VERSION` in `_app.py` AND `vscode-extension/package.json`
- After rebuilding the VSIX: update `_VSCODE_EXT_VERSION` in `_app.py`
- IPC hook may be stripped by `uv run` → `_find_vscode_ipc_hook()` walks parent process env
- Tunnel path: extension must configure `remote.portsAttributes` to make port public/silent

### Port & URL Construction

- Always use `localhost` (not `127.0.0.1`) so VS Code port-forwarding works
- Port default: `8123`; CLI: `--port` flag; `view()`: `port=` kwarg
- Compare mode URLs include `?compare_sid=...&compare_sids=...`
- Overlay URLs include `?overlay_sid=...`
- Mosaic/qMRI state is in the JS (not URL params); no server changes needed for those

### Julia-Specific Constraints

- GIL conflicts: never run server in-process when `_is_julia_env()` is True
- `_view_julia()` starts a detached subprocess running `arrayview_server` CLI tool
- Array data is serialized to a temp `.npy` file and loaded by the subprocess
- Julia arrays use PythonCall → numpy conversion before any arrayview API call
- No interactive stdin in subprocesses; all params must be encoded in CLI flags or files

### Jupyter-Specific Constraints

- `_in_jupyter()` returns True for ipykernel (VS Code notebook, JupyterLab, classic Notebook)
- Julia's IJulia kernel is NOT ipykernel → `_in_jupyter()` returns False in Julia notebooks (use Julia path instead)
- `inline=True` → returns `IPython.display.IFrame`; caller must return it from the cell for display
- Port reuse: calling `view()` multiple times reuses same port and server if already running
- `window='native'` override still works in Jupyter; viewer opens as separate window

## Step-by-Step Checklist

When implementing a new feature, ask:

1. **Does it touch server startup, teardown, or port selection?**
   - Test CLI (`uv run arrayview file.npy`) — server must start and terminate cleanly
   - Test Python script (`python script.py`) — daemon thread must not orphan
   - Test Jupyter cell — repeated calls must not fail on port-already-in-use
   - Check Julia path — `_view_julia()` must pass any new params via CLI flag or file

2. **Does it touch browser/display opening?**
   - Test local native window (macOS) — `pywebview` opens correctly
   - Test local system browser — `open http://localhost:8123` path works
   - Test VS Code terminal — Simple Browser opens via extension signal file
   - Test VS Code tunnel — route reaches client-side Simple Browser, not remote host browser
   - Verify `_ensure_vscode_extension()` installs/updates if VSIX changed

3. **Does it touch URL construction or query params?**
   - Verify `localhost` (not `127.0.0.1`)
   - Verify all modes still get the right params (compare, overlay, vectorfield)

4. **Does it touch environment detection?**
   - Check the detection order in `view()`: Julia → Jupyter → VS Code remote → VS Code terminal → local
   - Any new detection must not break the fallback chain for paths it shouldn't match
   - `_find_vscode_ipc_hook()` walks up to 12 parent processes — new subprocess wrappers may need the same treatment

5. **Does it change the VS Code extension?**
   - Rebuild VSIX: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`
   - Bump `_VSCODE_EXT_VERSION` in `_app.py` and `vscode-extension/package.json` together
   - Test install in a fresh VS Code profile

## Validation Matrix

After implementing, run through this matrix manually or in CI:

| Check | CLI | Python script | Jupyter | Julia | VS Code local | VS Code tunnel |
|-------|-----|--------------|---------|-------|--------------|----------------|
| Server starts | ✓? | ✓? | ✓? | ✓? | ✓? | ✓? |
| Array loads & renders | ✓? | ✓? | ✓? | ✓? | ✓? | ✓? |
| Display opens (window/browser/inline) | ✓? | ✓? | inline IFrame | browser | Simple Browser | Simple Browser on client |
| Ctrl-C / kernel stop cleans up | ✓? | ✓? | ✓? | ✓? | ✓? | ✓? |
| Compare mode works (if file-based) | ✓? | N/A | N/A | N/A | ✓? | ✓? |

Quick automated checks:
```bash
uv run pytest tests/test_api.py -x          # API contract
uv run pytest tests/test_cli.py -x          # CLI entry point
uv run python -c "from arrayview import view; import numpy as np; view(np.zeros((10,10)))"
```

## tmux and VS Code Terminal Detection

tmux is the single most common reason `_find_vscode_ipc_hook()` fails even though the user IS in a VS Code terminal.

### Why the ancestor-walk fails inside tmux

Normal process tree (no tmux):
```
VS Code terminal shell (has VSCODE_IPC_HOOK_CLI)
  └─ uv run python / arrayview   ← walks up, finds it ✓
```

Process tree inside tmux:
```
VS Code terminal shell (has VSCODE_IPC_HOOK_CLI)
  └─ tmux (client process — also has VSCODE_IPC_HOOK_CLI)
       ↕ socket IPC (not parent/child)
tmux-server (independent daemon — does NOT have VSCODE_IPC_HOOK_CLI)
  └─ pane shell
       └─ uv run python / arrayview   ← walks up through tmux-server, never finds it ✗
```

The arrayview process's parent chain goes through `tmux-server`, which is an independent daemon that was started at some point and does NOT inherit the VS Code terminal's environment.

### Why `tmux show-environment` fails

`tmux show-environment VSCODE_IPC_HOOK_CLI` queries tmux's *session* environment. tmux only tracks variables listed in its `update-environment` option. The default list is:
```
DISPLAY SSH_ASKPASS SSH_AUTH_SOCK SSH_AGENT_PID SSH_CONNECTION WINDOWID XAUTHORITY
```
`VSCODE_IPC_HOOK_CLI` is **not** in the default list, so it is never copied into tmux's session environment. The command returns `-VSCODE_IPC_HOOK_CLI` (meaning unset) even when every client has it.

### Why `#{client_pid}` alone is unreliable

`tmux display-message -p '#{client_pid}'` returns **one** client PID (the "current" client). This fails when:
- Multiple clients are attached to the session (e.g., shared session, or user has the session open in both VS Code and a regular terminal)
- A session was created outside VS Code and then attached from VS Code

### Correct approach: enumerate ALL clients for the current session

```python
# Get current session ID
session_id = subprocess.run(["tmux", "display-message", "-p", "#{session_id}"], ...).stdout.strip()

# List ALL clients for this session
client_pids = subprocess.run(
    ["tmux", "list-clients", "-t", session_id, "-F", "#{client_pid}"], ...
).stdout.strip().splitlines()

# Check each client's environment
for pid_str in client_pids:
    val = _ipc_from_pid(int(pid_str))
    if val and os.path.exists(val):
        return val  # found VSCODE_IPC_HOOK_CLI in one of the VS Code clients ✓
```

`list-clients -t <session_id>` scopes to the current session only, so we don't accidentally pick up a `VSCODE_IPC_HOOK_CLI` from a completely different VS Code window on another session.

### Detection order in `_find_vscode_ipc_hook()` (when TERM_PROGRAM=tmux)

1. `tmux show-environment VSCODE_IPC_HOOK_CLI` — cheap, works if user set `update-environment`
2. `tmux list-clients -t <session_id> -F '#{client_pid}'` → `ps ewwww` each — robust, handles all cases

### What `--diagnose` should show when working

```json
"detection": {
  "in_vscode_terminal": true,
  "vscode_ipc_hook_recovered": "/var/folders/.../vscode-ipc-xxx.sock"
}
```

If `vscode_ipc_hook_recovered` is `null` despite being in VS Code+tmux, the strategies above both failed. Check:
- Is the tmux session detached (no clients attached)?
- Is `ps ewwww -p <client_pid>` working on this OS? (Some Linux distros restrict it)
- Is the IPC socket path valid (does the `.sock` file exist)?

### Red flags for tmux detection

- "I fixed it with `#{client_pid}`" → only works with one client; use `list-clients` instead
- "I fixed it with `show-environment`" → only works if user customised `update-environment`; both strategies needed
- "It works for me but not for the user" → check if they have multiple clients or a session created before VS Code



- "I changed how the server starts but only tested CLI" → test all paths
- "The feature works locally but not in tunnel" → check `_is_vscode_remote()` path and port exposure
- "I used `127.0.0.1` in the URL" → use `localhost` so VS Code port-forwarding intercepts it
- "Julia works but I only tested the Python path" → Julia always uses subprocess — test `_view_julia()` explicitly
- "I bumped the extension version in package.json but not `_VSCODE_EXT_VERSION`" → they must stay in sync
- "The server starts but leaves an orphan process after Ctrl-C" → check `_shutdown_event`, signal handlers, and subprocess reaping
