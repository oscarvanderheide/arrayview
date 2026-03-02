# ArrayView — Tunnel Auto-Open Plan

## Goal
Make `arrayview` auto-open in VS Code's **Simple Browser** tab when running over a
VS Code tunnel (remote Linux ← macOS client). No new VS Code windows. No manual
cmd-clicking URLs. Jupyter inline should also work over the tunnel.

## What's Already Done (in this branch)

### 1. Companion VS Code Extension (`vscode-extension/`)
- **`arrayview-opener`** v0.0.3 — **file-watching IPC** + URI handler
- **Primary mechanism** (tunnel/remote): the extension watches `~/.arrayview/open-request.json`.
  When the Python backend writes a URL to that file, the extension reads it, resolves
  via `vscode.env.asExternalUri` (port forwarding), and calls `simpleBrowser.show`.
- **Secondary mechanism** (local/SSH): URI handler `vscode://arrayview.arrayview-opener/open?url=<encoded>`
  still works on setups where `code --open-url` supports `vscode://` URIs.
- Has `extensionKind: ["workspace", "ui"]` so it can be installed on the **remote** side
  (unlike v0.0.2 which was `["ui"]`-only and couldn't be installed via remote-cli `code`).
- `activationEvents: ["onStartupFinished", "onUri"]` — activates on startup to begin file-watching.
- Bundled as `src/arrayview/arrayview-opener.vsix` (included in the Python wheel)
- Auto-installed by `_ensure_vscode_extension()` on first use

### 2. Changes to `_app.py`

| Change | Why |
|--------|-----|
| All user-facing URLs → `localhost` instead of `127.0.0.1` | VS Code tunnel auto-forwards `localhost:PORT` but NOT `127.0.0.1:PORT`. This was causing ERR_CONNECTION_REFUSED. |
| `_is_vscode_remote()` rewritten | Old version had operator precedence bug and required `SSH_CONNECTION` which tunnels don't set. New version: on Linux, `VSCODE_IPC_HOOK_CLI` alone = remote. |
| `_find_code_cli()` — added `~/.vscode/cli/servers/*/server/bin/remote-cli/code` glob | The actual path on the remote was `~/.vscode/cli/servers/Stable-.../server/bin/remote-cli/code`, not `~/.vscode-server/...` |
| `_ensure_vscode_extension()` — version-aware + error-checking | Now checks `--show-versions` and force-reinstalls if version doesn't match `_VSCODE_EXT_VERSION`. **Also checks stdout for "Cannot install" errors** — the remote-cli `code` returns exit code 0 even on failure. |
| `_ensure_vscode_extension()` — passes IPC hook | `uv run` strips env vars; now explicitly passes `VSCODE_IPC_HOOK_CLI` in env for `code` subprocess calls |
| `_open_via_signal_file()` — **NEW** | Writes URL to `~/.arrayview/open-request.json` for the extension's file-watcher to pick up. This is the primary IPC mechanism for tunnels since `code --open-url` is not supported by the remote-cli. |
| `_open_browser()` — rewritten | For tunnel/remote: uses signal file as primary method. Falls back to `code --open-url` (checks for "Ignoring option" response), then `xdg-open`, then prints URL. |

### 3. Issues Fixed in v0.0.3

| Issue | Resolution |
|-------|------------|
| `code --open-url` not supported by remote-cli | Replaced with file-based IPC (signal file) |
| `code --install-extension` failing for `extensionKind: ["ui"]` | Changed to `["workspace", "ui"]` — extension now installs on remote |
| `_ensure_vscode_extension()` false positive | Now checks stdout for "Cannot install" error message |
| Extension not activating on install | Added `onStartupFinished` activation event; `fs.watch` begins immediately |

### 3. Remote Environment (from diagnostics)
```
VSCODE_IPC_HOOK_CLI=/run/user/1885/vscode-ipc-51d2e032-18f2-4062-87f0-61b5daca01a6.sock
TERM_PROGRAM=vscode
DISPLAY=:1  (misleading — no real display in tunnel)
which code → /nfs/rtfs03/storage/home/oheide/.vscode/cli/servers/Stable-.../server/bin/remote-cli/code
Home: /nfs/rtfs03/storage/home/oheide (NFS mount), ~ → /home/oheide (symlink)
SSH_CONNECTION is NOT set (tunnel, not SSH)
```

## Verification Results (Tested)

### Step 1: Extension installed & up-to-date ✅
```
$ code --list-extensions --show-versions | grep arrayview
arrayview.arrayview-opener@0.0.3
```

### Step 2: Detection functions ✅
```
Platform: linux
VSCODE_IPC_HOOK_CLI: /run/user/1885/vscode-ipc-...sock
TERM_PROGRAM: vscode
SSH_CONNECTION: NOT SET

is_vscode_remote(): True
find_code_cli(): /home/oheide/.vscode-server/bin/.../bin/remote-cli/code
find_vscode_ipc_hook(): /run/user/1885/vscode-ipc-...sock
Required ext version: 0.0.3
ensure_extension(): True
```

### Step 3: `code --open-url` NOT supported ⚠️ (expected for tunnel remote-cli)
```
$ code --open-url "http://localhost:8080"
Ignoring option 'open-url': not supported for code.
```
This is why v0.0.3 uses file-based IPC instead.

### Step 4: Signal file mechanism ✅
- `_open_via_signal_file()` writes `~/.arrayview/open-request.json`
- Extension's `fs.watch` detects the file and consumes it
- Signal file is deleted after processing

### Step 5: End-to-end `view()` ✅
```python
import numpy as np
from arrayview import view
view(np.random.rand(50, 50, 50))
```
- Server starts on port 8123
- Signal file written and consumed by extension
- Simple Browser opens with the viewer

### Step 6: CLI ✅
```
$ uv run arrayview small_array.npy --browser
Loaded small_array.npy with shape (10, 10, 10, 10) (0 MB)
Open http://localhost:8000/shell?...
```

## Known Issues / Potential Problems

1. **Extension reload after install**: After first install, VS Code may need a
   window reload (`Developer: Reload Window`) for the extension to activate and
   start file-watching. Subsequent sessions work automatically since the extension
   activates `onStartupFinished`.

2. **Port forwarding not working**: `asExternalUri` in the extension handles
   this. Using `localhost` (not `127.0.0.1`) in URLs is critical.

3. **`uv run` stripping env vars**: `_find_vscode_ipc_hook()` walks the process
   tree via `/proc/<pid>/environ` to recover `VSCODE_IPC_HOOK_CLI`. This is
   already implemented.

4. **NFS home vs symlink**: Home dir is `/nfs/rtfs03/storage/home/oheide` but
   `~` resolves to `/home/oheide`. Both are valid; `os.path.expanduser("~")` gives
   `/home/oheide` which works with the symlink.

## File Locations
- Python code: `src/arrayview/_app.py`
- Extension source: `vscode-extension/extension.js` + `vscode-extension/package.json`
- Bundled vsix: `src/arrayview/arrayview-opener.vsix`
- Extension version constant: `_VSCODE_EXT_VERSION` in `_app.py` (must match `package.json`)
- Signal file: `~/.arrayview/open-request.json` (written by Python, consumed by extension)

## Architecture: How the Tunnel Auto-Open Works

```
Python (_app.py)                    VS Code Extension (workspace side)
─────────────────                   ──────────────────────────────────
view(data) / arrayview CLI
  │
  ├─ Start uvicorn server on localhost:PORT
  │
  ├─ _ensure_vscode_extension()
  │    └─ code --install-extension .vsix --force
  │
  └─ _open_browser(url)
       └─ _open_via_signal_file(url)
            │
            └─ Write ~/.arrayview/open-request.json ──→ fs.watch detects file
                 {"url": "http://localhost:PORT/..."}       │
                                                            ├─ Read & delete file
                                                            ├─ asExternalUri(url) → port-forwarded URL
                                                            └─ simpleBrowser.show(resolvedUrl)
                                                                   │
                                                                   └─ Simple Browser tab opens
                                                                      (port-forwarded through tunnel)
```
