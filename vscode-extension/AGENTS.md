# VS Code Extension

Load the `vscode-simplebrowser` skill before touching anything here.

## Key Functions (`_vscode.py`)

- `_ensure_vscode_extension()` — installs VSIX only if version not on disk
- `_open_via_signal_file()` — signal-file IPC, extension polls and opens panel (network mode)
- `_open_direct_via_signal_file()` — direct webview mode (stdio, no network)
- `_open_direct_via_shm()` — shared memory IPC for zero-copy numpy arrays
- `_schedule_remote_open_retries()` — retry for tunnel/first-install latency

## Two Transport Modes

- **Network** (default): FastAPI + WebSocket, extension opens Simple Browser tab
- **Direct webview** (`--mode stdio`): extension spawns subprocess, PythonBridge bridges postMessage ↔ stdin/stdout

`_VSCODE_EXT_VERSION` in `_vscode.py` must match `vscode-extension/package.json`.
Rebuild: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`

## High-Risk

- Skip `--force` if correct version on disk; clean stale dirs first
- IPC hook recovery when env vars stripped by `uv run` or subprocess wrappers
- tmux: `VSCODE_IPC_HOOK_CLI` not inherited; must walk client PIDs
- Signal routing: local → per-window targeted file; remote/tunnel → shared fallback
- Direct webview: stdin/stdout binary protocol, subprocess lifecycle, PythonBridge error handling
- Shared memory: POSIX SHM with resource tracker cleanup
