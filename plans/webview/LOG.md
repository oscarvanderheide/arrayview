# Webview Message Passing Debug Log

## Problem Statement
`uv run arrayview medium_array.npy` in VS Code tunnel: opens webview tab the first time, but subsequent runs (after closing tab + Ctrl+C) do nothing.

## Root Cause Found (2026-03-31)

### Extension log for first run (works):
```
DISPATCH: file=open-request-v0900.json mode=direct hasFilepath=true ...
SIGNAL-DATA: mode=direct filepath=...medium_array.npy
PYTHON: spawning .../python3 -m arrayview --mode stdio .../medium_array.npy
SIGNAL: missing url          <-- v0800 compat duplicate being processed
SESSION READY: sid=...
DIRECT: opened "ArrayView: medium_array.npy"
```

### Extension log for second run (broken):
```
PYTHON: exited with code null   <-- user closed tab + Ctrl+C
SIGNAL: missing url              <-- NO DISPATCH line!
SIGNAL: missing url
```

### Analysis
1. First run: v0900 signal (mode=direct) is processed correctly. v0800 compat duplicate also fires, falls through to "missing url" (harmless).
2. User closes tab, presses Ctrl+C -> Python subprocess exits (`code null` = SIGTERM)
3. Second run: signal files are written by Python, but NO `DISPATCH:` log appears, meaning the signals are being consumed before reaching `processSignalData`.

### Key insight: the v0800 compat signal from the FIRST run
Python writes to BOTH `open-request-v0900.json` AND `open-request-v0800.json` (same data). On the first run:
- Tick 1: v0900 is claimed and processed (direct mode -> success)
- Tick 2: v0800 is claimed and processed (direct mode, but duplicate -> "missing url" because it passes the direct check but the Python process already exited, OR it's a different issue)

Wait -- looking again at the log: `SIGNAL: missing url` at 10:12:12.781 happens DURING the first run's processing (between PYTHON spawn and SESSION READY). This is the v0800 being processed on the SAME tick? No -- `tryOpenSignalFile` processes one signal per tick and returns.

Actually the v0800 `SIGNAL: missing url` at 10:12:12.781 has NO `DISPATCH:` line either! So this is coming from a DIFFERENT code path or an OLD extension host.

### Root cause hypothesis
There may be TWO signal file consumers:
1. The fs.watch handler
2. The setInterval polling handler

Both call `tryOpenSignalFile()`. The `isProcessingSignal` flag prevents re-entry WITHIN `processSignalData`, but the claim (rename) happens BEFORE `processSignalData` is called. Two concurrent `tryOpenSignalFile` calls could each claim different files.

Actually no -- looking at the code flow: `tryOpenSignalFile` iterates through candidates, tries to rename (claim) each one. If rename succeeds, it processes that file. If another call already claimed it, `rename` throws and `continue` moves to the next candidate.

The REAL issue: the v0800 compat signal is being claimed and processed by a SECOND `tryOpenSignalFile` invocation that runs with the OLD in-memory code (no DISPATCH logging). This happens because `fs.watch` can fire multiple times for a single file write, and each fires `tryOpenSignalFile`.

### Actual fix needed
The v0800 compat signal duplicates are harmful. The second signal gets processed while the first is still in-flight (the `isProcessingSignal` flag isn't set yet when the second call starts because the first hasn't reached `processSignalData` yet -- it's still in the claiming loop).

**The real fix**: Stop writing to the v0800 compat signal file. Only write to v0900. The v0800 compat path was for older extension versions which are no longer relevant since we control the extension install.

OR: the second run fails because the v0800 from the SECOND run gets claimed by a concurrent `tryOpenSignalFile` that started before `isProcessingSignal` was set by the first one.

## Previous issues resolved
1. **System python not found** (initial): Extension spawned `/usr/bin/python3` which didn't have arrayview. Fixed by passing `pythonPath` (sys.executable) in the signal file.
2. **Extension not reloading**: `code --install-extension --force` updates files on disk but the extension host in a tunnel doesn't auto-restart. Need "Developer: Restart Extension Host" specifically.

## Fix: v0800 compat duplicate causing isProcessingSignal deadlock

**Root cause confirmed**: Python writes to both `v0900` and `v0800` signal files (same data). The v0800 duplicate gets processed after the first direct webview completes, triggering a SECOND `openDirectWebview` that spawns another Python subprocess. This keeps `isProcessingSignal = true` for up to 30 seconds (timeout). During that window, any new signals from a second `arrayview` invocation are silently dropped.

**Fix applied**:
1. **Python side**: `_open_direct_via_signal_file()` and `_open_direct_via_shm()` now pass `skip_compat=True` to `_write_vscode_signal()`, so direct-mode signals only write to `v0900` (no v0800 duplicate).
2. **Extension side**: After detecting a direct-mode signal, immediately delete any compat signal files (`v0800`, `v0400`) before spawning the subprocess. This prevents duplicates even if old Python code writes compat files.

## Fix: extension not loading due to version mismatch (2026-03-31, session 2)

**Symptom**: `uv run arrayview medium_array.npy` writes signal file, but nothing happens. Extension log shows no `ACTIVATE` since 10:50. Extension hosts running (PIDs 28784, 4052896) but not loading the extension.

**Root cause**: `_VSCODE_EXT_VERSION` in `_vscode.py` was `"0.10.0"` but the bundled VSIX and `vscode-extension/package.json` are at `"0.10.2"`. This caused a cascade:
1. `_ensure_vscode_extension()` calls `_remove_old_extension_versions("0.10.0")`, which deletes the `arrayview.arrayview-opener-0.10.2/` directory.
2. `_extension_on_disk("0.10.0")` finds the v0.10.0 directory and skips reinstall.
3. But `extensions.json` (VS Code's extension registry) still points to the v0.10.2 path.
4. VS Code extension host looks for `arrayview.arrayview-opener-0.10.2/`, can't find it, silently skips loading → no ACTIVATE, no signal handling.

**Evidence**:
- `extensions.json` referenced `arrayview.arrayview-opener-0.10.2` but only `0.10.0` directory existed
- `code --list-extensions --show-versions` showed `@0.10.2` (from metadata) but directory was named `0.10.0`
- No ACTIVATE log entries after 10:50 despite two running extension hosts
- `SIGNAL: missing url` entries in log came from a PREVIOUS (now-dead) extension host whose fs.watch/setInterval leaked past deactivation

**Fix**:
1. Updated `_VSCODE_EXT_VERSION` to `"0.10.2"` in `_vscode.py`
2. Uninstalled + reinstalled extension to sync directory name with metadata
3. Restart extension host needed to load the extension

## Multi-window targeting issue (2026-03-31, session 3)

**Symptom**: When multiple VS Code tunnel windows are open to the same remote (different directories), `arrayview` opens the webview tab in the wrong window.

**Analysis**:
In tunnel/remote mode, the original code skips hookTag matching (`if ipc_hook and not _is_vscode_remote()` — line 642) and falls back to PID ancestry matching. Two potential problems:

1. **hookTag matching was disabled for remotes**: The comment says "terminal and extension host have different IPC hooks in tunnel mode" — but this may not be true. In SSH remotes and possibly tunnels, both the terminal and extension host may share the same `VSCODE_IPC_HOOK_CLI` from the same VS Code server instance. If so, hookTag is the most reliable targeting method.

2. **PID ancestry may be ambiguous**: If the tunnel server's process tree doesn't clearly separate per-window subtrees, PID ancestry matching picks the wrong window (or falls to broadcast, where any window can claim the signal).

**Root cause confirmed via diagnostics**:
1. Both extension hosts have IDENTICAL ppids `[3342979, 3342975, 22342, ...]` — both are direct children of the same `node` process. PID ancestry matching gives identical scores.
2. In a tunnel, all terminals share the same `VSCODE_IPC_HOOK_CLI` socket (the tunnel server's CLI socket), so hookTag is the same for both terminals → always targets the same window.

**Fix: EnvironmentVariableCollection**:
- **Extension side**: Uses `context.environmentVariableCollection.replace('ARRAYVIEW_WINDOW_ID', windowId)` to inject a window-specific env var into all terminals opened in that window. Each window has a unique `windowId` (hookTag or PID).
- **Python side**: New `_find_arrayview_window_id()` reads `ARRAYVIEW_WINDOW_ID` from direct env or ancestor `/proc/<pid>/environ` (handles `uv run` env stripping). Used as primary targeting method for remote/tunnel, with PID ancestry as fallback.
- Rebuilt VSIX, force-reinstalled extension, cleaned up stale window registrations.

**Result**: Fix confirmed working — each window now correctly targets its own webview tab.

## Fix: av.view() in tunnel using ports instead of direct webview (2026-03-31, session 3)

**Symptom**: `av.view(x)` from a Python script in a VS Code tunnel terminal shows "Remote tunnel session on port 8123" message and uses port-based approach instead of direct webview.

**Root cause**: A stale server running on port 8123 (from a previous CLI session) caused `_server_alive(port)` at line 836 to return True, taking the URL-based path before reaching the direct webview path at line 928.

**Fix**:
1. Moved the non-Jupyter tunnel direct webview check ABOVE the `_server_alive` check. Tunnel sessions now always use direct webview regardless of stale servers.
2. Changed the `_server_alive` path to only activate for non-remote sessions (`not _is_vscode_remote()`).
3. Jupyter in tunnel: changed from "webview tab + message" to inline IFrame mode. VS Code tunnel auto-forwards ports, so IFrames work for the tunnel owner. Added `_configure_vscode_port_preview(port)` to set the port as silent.

## Changes made so far
- `_vscode.py`: `_open_direct_via_signal_file()` now includes `pythonPath: sys.executable`
- `_vscode.py`: New `_open_direct_via_shm()` for passing arrays via shared memory
- `_launcher.py`: `view()` tunnel paths use `_open_direct_via_shm()` instead of temp files
- `_launcher.py`: Added `--shm-name/--shm-shape/--shm-dtype/--name` CLI args for stdio mode
- `extension.js`: `PythonBridge` accepts `pythonPath` and `shmParams`
- `extension.js`: `processSignalData` handles `data.shm` for shared memory mode
- `extension.js`: Debug logging (`DISPATCH:`, `SIGNAL-DATA:`)
